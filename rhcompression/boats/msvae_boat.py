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
        self.weight_adv_patch = float(hps.get('w_adv_patch', 0.8))

        self.models['dist_ema'] = StateEMA()

    def d_step_calc_losses(self, batch):

        if self.global_step() >= self.start_adv:

            batch = move_to_device(batch, self.device)
            x = batch['gt']

            with torch.no_grad():
                output = self.models['net'](x, mode='full') 

            d_real_patch = self.models['patch_critic'](x)
            d_fake_patch = self.models['patch_critic'](output['x_sample_sg'].detach())
            d_loss_patch = self.losses['patch_critic'](
                {'real': d_real_patch, 'fake': d_fake_patch, **batch}
            )

            d_real_image = self.models['image_critic'](x)
            d_fake_image = self.models['image_critic'](output['x_sample_sg'].detach())
            d_loss_image = self.losses['critic'](
                {'real': d_real_image, 'fake': d_fake_image, **batch}
            )

            d_loss = self.weight_adv_patch * d_loss_patch + (1 - self.weight_adv_patch) * d_loss_image

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

        # Adversarial Loss
        if self.global_step() >= self.start_adv:
            g_real_patch = self.models['patch_critic'](output['x_sample_sg'])
            l_adv_patch = self.losses['critic']({'real': g_real_patch, 'fake': None, **batch})
            g_real_image = self.models['image_critic'](output['x_sample_sg'])
            l_adv_image = self.losses['critic']({'real': g_real_image, 'fake': None, **batch})
            l_adv = self.weight_adv_patch * l_adv_patch + (1 - self.weight_adv_patch) * l_adv_image
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

