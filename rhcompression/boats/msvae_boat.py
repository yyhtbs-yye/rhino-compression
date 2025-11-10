import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcompression.boats.sdvae_boat import SDVAEBoat
from rhcompression.boats.helpers.state_ema import StateEMA
from rhtrain.utils.ddp_utils import move_to_device

# MeanStay VAE (Mainstay VAE/AnchorVAE)
class MSVAEBoat(SDVAEBoat):
    def __init__(self, config=None):
        super().__init__(config=config or {})
        cfg = config or {}
        hps = cfg.get("boat", {}).get("hyperparameters", {})

        # VA-VAE specifics
        self.w_lpf     = float(hps.get('w_lpf', 0.5))       # whyper in the paper
        self.w_ms      = float(hps.get('w_ms', 0.5))       # whyper in the paper
        self.f_lpf     = float(hps.get('f_lpf', 0.5))
        self.f_hpf     = float(hps.get('f_hpf', 0.5))

        self.models['dist_ema'] = StateEMA()

    def _lowpass_loss(self, x_hat, x, mag):
        """
        x_hat, x: (B, C, H, W)
        mag: float or tensor of shape (B,), each sample its own cutoff.
        """
        assert x_hat.shape == x.shape, "x_hat and x must have the same shape"
        B, C, H, W = x.shape

        # mag to tensor of shape (B,)
        if isinstance(mag, (float, int)):
            mag = torch.full((B,), float(mag), device=x.device, dtype=x.dtype)
        else:
            mag = torch.as_tensor(mag, device=x.device, dtype=x.dtype)
            assert mag.shape[0] == B, "mag must have length B"

        # clamp to (0, 1]
        mag = mag.clamp(1e-3, 1.0)

        # FFT over spatial dims
        X_hat = torch.fft.fftn(x_hat, dim=(-2, -1))
        X     = torch.fft.fftn(x,     dim=(-2, -1))

        # radius map (H, W)
        yy = torch.arange(H, device=x.device) - H // 2
        xx = torch.arange(W, device=x.device) - W // 2
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        radius = torch.sqrt(yy**2 + xx**2)              # (H, W)

        r_max = radius.max()
        radius = radius.view(1, 1, H, W)                # (1, 1, H, W)

        # per-sample cutoff: (B, 1, 1, 1)
        cutoff = (mag.view(B, 1, 1, 1) * r_max)

        # broadcast to (B, 1, H, W)
        mask = (radius <= cutoff).float()               # (B, 1, H, W)

        # shift mask to match unshifted FFT layout
        mask = torch.fft.ifftshift(mask, dim=(-2, -1))

        # apply low-pass filter
        X_hat_lp = X_hat * mask
        X_lp     = X     * mask

        # back to spatial domain (take real part)
        x_hat_lp = torch.fft.ifftn(X_hat_lp, dim=(-2, -1)).real
        x_lp     = torch.fft.ifftn(X_lp,     dim=(-2, -1)).real

        # MSE in low-frequency space
        loss = F.mse_loss(x_hat_lp, x_lp)

        return loss

    # -------- D step wiring (real vs recon) --------
    def d_step_calc_losses(self, batch):

        if self.global_step() >= self.start_adv:

            batch = move_to_device(batch, self.device)
            x = batch['gt']

            with torch.no_grad():
                output = self.models['net'](x, mode='full') 

            d_real = self.models['critic'](x)
            d_fake = self.models['critic'](output['x_sample_sg'].detach())

            train_out = {'real': d_real, 'fake': d_fake, **batch}

            d_loss = self.losses['critic'](train_out)

            w_adv = self.max_weight_adv * float(self.adv_fadein(self.global_step()))

            return {'d_loss': d_loss * w_adv}
        else:
            return {'d_loss': torch.zeros((), device=self.device)}

    # ---------- Train ---------- 
    def g_step_calc_losses(self, batch):

        x = batch['gt']

        output = self.models['net'](x, mode='full')
        
        l_ms = self.losses['pixel_loss'](output['x_mean'], x)

        l_lpf = self._lowpass_loss(output['x_sample'], x, mag=0.75)

        w_adv = self.max_weight_adv * float(self.adv_fadein(self.global_step()))

        if self.global_step() >= self.start_adv:
            d_fake_for_g = self.models['critic'](output['x_sample_sg'])
            train_out = {'real': d_fake_for_g, 'fake': None, **batch}
            l_adv = self.losses['critic'](train_out)
        else:
            l_adv = torch.zeros((), device=self.device)

        w_lpips = self.max_weight_lpips * float(self.lpips_fadein(self.global_step()))
        l_lpips = (self.losses['lpips_loss'](output['x_mean'], x).mean() * 0.5
                   if ('lpips_loss' in self.losses and w_lpips > 1e-6)
                   else torch.zeros((), device=self.device))

        l_kl = output['q'].kl().mean()

        g_loss = (
            self.w_ms * l_ms 
            + w_adv * l_adv 
            + w_lpips * l_lpips 
            + self.beta_kl * l_kl
            + l_lpf * self.w_lpf
        )

        return {
            'g_loss': g_loss,
            'w_adv': torch.tensor(w_adv),
            'l_adv': l_adv.detach(),
            'w_lpips': torch.tensor(w_lpips),
            'l_lpips': l_lpips.detach(),
            'l_kl': l_kl.detach(),
            'l_ms': l_ms.detach(),
            'l_lpf': l_lpf.detach(),
        }

