import math
import torch
import torch.nn.functional as F

from rhtrain.utils.ddp_utils import move_to_device, get_raw_module
from rhcore.boats.base_boat import BaseBoat

class VQVAEBoat(BaseBoat):
    """
    VQ-VAE training wired for DDP:
      - self.models['net']: AutoEncoder (encode/decode only) via forward(...)
      - self.models['vq'] : VectorQuantizer (codebook ops only) via forward(...)
    """

    def __init__(self, config={}):
        super().__init__(config=config)
        hps = config.get('hyperparameters', {})

        self.lambda_image  = float(hps.get('lambda_image', 1.0))
        self.lambda_image_bs  = float(hps.get('lambda_image_bs', 2.0))
        self.lambda_codebook = float(hps.get('lambda_codebook', 1.0))
        self.lambda_commit = float(hps.get('lambda_commit', 1.0))
        self.lambda_lpips  = float(hps.get('lambda_lpips', 1.0))

        # anneal_base ** math.round((step - annealing_start) /anneal_speed)
        self.anneal_start = 6000
        self.anneal_base = 0.99
        self.anneal_speed = 5

    def predict(self, x):
        h = self.models['net'](x, mode='encode', clamp=True)
        x_hat_bs = self.models['net'](h, mode='decode', clamp=True)
        c = self.models['vq'](h, mode='encode')
        h_hat = self.models['vq'](c, mode='decode')
        x_hat = self.models['net'](h_hat, mode='decode', clamp=True)

        return x_hat, x_hat_bs

    def training_calc_losses(self, batch):
        batch = move_to_device(batch, self.device)
        gt = batch['gt']

        z_e = self.models['net'](gt, mode='encode', clamp=False)   
        
        c = self.models['vq'](z_e, mode='encode')

        z_q = self.models['vq'](c, mode='decode')

        z_q_st = z_e + (z_q - z_e).detach()            # straight-through

        z_e_sg = z_e.detach()
        z_q_sg = z_q.detach()

        x_hat = self.models['net'](z_q_st, mode='decode', clamp=False)
        x_hat_bs = self.models['net'](z_e, mode='decode', clamp=False) # Teacher Force Bootstrap

        # Reconstruction
        l_image = self.losses['pixel_loss'](x_hat, gt) 
        l_image_bs = self.losses['pixel_loss'](x_hat_bs, gt)
        l_codebook = self.losses['codebook_loss'](z_e_sg, z_q)
        l_commit = self.losses['commit_loss'](z_e, z_q_sg)

        # Perceptual 
        l_lpips = torch.zeros((), device=self.device)
        if self.lambda_lpips > 0 and 'lpips_loss' in self.losses:
            lp = self.losses['lpips_loss'](x_hat, gt)
            l_lpips = (lp.mean() if lp.ndim > 0 else lp)

        weight_bs = self.anneal_base ** round(max(0, self.global_step() - self.anneal_start) /self.anneal_speed)

        total_loss = l_image * self.lambda_image + l_image_bs * weight_bs \
            + l_codebook * self.lambda_codebook + l_commit * self.lambda_commit + l_lpips * self.lambda_lpips

        with torch.no_grad():
            K = get_raw_module(self.models['vq']).vocab_size  # or however you expose it
            hist = torch.bincount(c.view(-1), minlength=K).float()
            usage = (hist > 0).float().mean()         # fraction of codes used
            p = (hist / hist.sum()).clamp_min(1e-12)
            perplexity = torch.exp(-(p * p.log()).sum())

        # Return losses + metrics (your logger can pick these up)
        return {
            'total_loss': total_loss,
            'l_image': l_image.detach(),
            'l_image_bs': l_image_bs.detach(),
            'l_codebook': l_codebook.detach(),
            'l_commit': l_commit.detach(),
            'l_lpips': l_lpips.detach(),
            'perplexity': perplexity,
            'usage': usage,
            'weight_bs': torch.tensor(weight_bs),
        }

    # -------- Validation: reconstruct inputs --------
    def validation_step(self, batch, batch_idx, epoch):
        if batch_idx == 0:
            self._reset_metrics()

        batch = move_to_device(batch, self.device)
        gt = batch['gt']

        with torch.no_grad():
            x_hat, x_hat_bs = self.predict(gt)
            valid_output = {'preds': x_hat, 'targets': gt}
            metrics = self._calc_metrics(valid_output)
            named_images = {'gt': gt, 'recon': x_hat, 'decoded': x_hat_bs}

        return metrics, named_images
