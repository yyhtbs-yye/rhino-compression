import torch
import torch.nn as nn
import torch.nn.functional as F

from rhtrain.utils.ddp_utils import move_to_device

from rhadversarial.boats.base_gan_boat import BaseGANBoat

class VARVQGANBoat(BaseGANBoat):
    """
    Multi-scale VQ-GAN tokenizer training inside BaseGANBoat, with:
      G loss = λ_image * recon + λ_latent * latent + λ_lpips * lpips + λ_vq * vq + λ_adv * adv
      D loss = adversarial( real vs recon ), optional extra regularizers via self.losses
    Models and losses are provided by BaseBoat from config.
    """

    def __init__(self, config={}):
        super().__init__(config=config)
        hps = config.get('hyperparameters', {})

        # scalar weights
        self.lambda_image  = float(hps.get('lambda_image', 1.0))
        self.lambda_latent = float(hps.get('lambda_latent', 1.0))
        self.lambda_lpips  = float(hps.get('lambda_lpips', 1.0))
        self.lambda_vq     = float(hps.get('lambda_vq', 1.0))

        # optional discriminator regularizers (e.g. 'r1', 'gp') — keys in self.losses
        self.d_reg_keys = hps.get('d_regularizers', [])

    # -------- AE forward helpers (use model.forward so DDP sees all ops) --------
    def _reconstruct(self, x, detach_for_d=False):
        """
        Encode -> quantize (STE) -> decode via model.forward(...).
        Returns (recon, z_pre, z_quant, vq_loss).
        """
        net = self.models['net']  # expected to be VARVQGAN

        out = net(x, ret_usages=False, clamp=False)
        recon   = out['recon']
        z_pre   = out['z_pre']
        z_quant = out['z_quant']
        vq_loss = out['vq_loss']

        if detach_for_d:
            recon = recon.detach()

        return recon, z_pre, z_quant, vq_loss

    # -------- D step wiring (real vs recon) --------
    def d_step_calc_losses(self, batch):
        batch = move_to_device(batch, self.device)
        gt = batch['gt']

        with torch.no_grad():
            recon, _, _, _ = self._reconstruct(gt, detach_for_d=True)

        d_real = self.models['critic'](gt)
        d_fake = self.models['critic'](recon)

        train_out = {'real': d_real, 'fake': d_fake, **batch}
        d_loss = self.losses['critic'](train_out)

        # optional regularizers (e.g., R1) supplied as callable losses in self.losses
        for k in self.d_reg_keys:
            d_loss = d_loss + self.losses[k]({'d_real': d_real, 'image_real': gt, **batch})

        return {
            'd_loss': d_loss,
            'd_real': d_real.mean().detach(),
            'd_fake': d_fake.mean().detach(),
        }

    # -------- G step wiring (compound objective) --------
    def g_step_calc_losses(self, batch):
        
        batch = move_to_device(batch, self.device)
        
        gt = batch['gt']

        recon, z_pre, z_quant, vq_loss = self._reconstruct(gt, detach_for_d=False)

        # reconstruction loss
        l_image = self.losses['pixel_loss'](recon, gt) * self.lambda_image

        # latent alignment loss (z_quant vs z_pre)
        l_latent = self.losses['latent_loss'](z_quant, z_pre) * self.lambda_latent

        # perceptual loss (if configured)
        l_lpips = torch.zeros((), device=self.device)
        if self.lambda_lpips > 0 and 'lpips_loss' in self.losses:
            lp = self.losses['lpips_loss'](recon, gt)
            l_lpips = (lp.mean() if lp.ndim > 0 else lp) * self.lambda_lpips

        # adversarial loss for generator: feed FAKE logits only
        d_fake_for_g = self.models['critic'](recon)
        adv_in = {'real': d_fake_for_g, 'fake': None,  **batch}
        l_adv = self.losses['critic'](adv_in) * self.adversarial_weight

        # codebook/commitment loss from quantizer
        l_vq = vq_loss * self.lambda_vq

        g_loss = l_image + l_latent + l_lpips + l_adv + l_vq

        return {
            'g_loss': g_loss,
            'l_image': l_image.detach(),
            'l_latent': l_latent.detach(),
            'l_lpips': l_lpips.detach(),
            'l_adv': l_adv.detach(),
            'l_vq': l_vq.detach(),
        }

    # -------- Validation: reconstruct inputs (autoencoder), not noise sampling --------
    def validation_step(self, batch, batch_idx, epoch):
        if batch_idx == 0:
            self._reset_metrics()

        batch = move_to_device(batch, self.device)
        gt = batch['gt']

        with torch.no_grad():
            recon, _, _, _ = self._reconstruct(gt, detach_for_d=True)
            valid_output = {'preds': recon, 'targets': gt}
            metrics = self._calc_metrics(valid_output)
            named_images = {'gt': gt, 'recon': recon}

        return metrics, named_images
