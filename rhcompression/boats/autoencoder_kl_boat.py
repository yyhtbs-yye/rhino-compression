import torch
from rhtrain.utils.ddp_utils import move_to_device
from rhcore.boats.base_boat import BaseBoat
from rhcore.utils.build_components import build_module

class AutoencoderKLBoat(BaseBoat):
    def __init__(self, config=None):
        super().__init__(config=config or {})
        hps  = config['boat'].get('hyperparameters', {})

        self.lambda_image   = float(hps.get('lambda_image', 1.0))
        self.beta_kl        = float(hps.get('beta_kl', hps.get('lambda_kl', 1e-6)))
        self.max_weight_lpips    = float(hps.get('max_weight_lpips', 1.0))
        self.use_mode_eval  = bool(hps.get('use_mean_at_eval', True))

        # Annealers are optional; default to constant schedules
        self.lpips_fadein = build_module(config['boat']['lpips_fadein']) if 'lpips_fadein' in config['boat'] else (lambda *_: 0.0)

    # ---------- Inference ----------
    def predict(self, x):
        sm = 'mode' if self.use_mode_eval else 'random'
        x_hat, _ = self.models['net'](x, mode='full', sample_method=sm)
        return torch.clamp(x_hat, -1.0, 1.0)

    # ---------- Train ----------
    def training_calc_losses(self, batch):
        x = move_to_device(batch, self.device)['gt']
        x_hat, q = self.models['net'](x, mode='full', sample_method='random')

        l_img = self.losses['pixel_loss'](x_hat, x)

        w_lpips = self.max_weight_lpips * float(self.lpips_fadein(self.global_step()))
        
        l_lpips = (self.losses['lpips_loss'](x_hat, x).mean()
                   if ('lpips_loss' in self.losses and w_lpips > 1e-6)
                   else torch.zeros((), device=self.device))

        l_kl = (q.kl().mean() if hasattr(q, 'kl') else torch.zeros((), device=self.device))

        total = self.lambda_image * l_img + w_lpips * l_lpips + self.beta_kl * l_kl

        return {
            'total_loss': total,
            'l_image': l_img.detach(),
            'l_lpips': l_lpips.detach(),
            'l_kl': l_kl.detach(),
            'w_lpips': torch.tensor(w_lpips),
        }

    # ---------- Validation ----------
    def validation_step(self, batch, batch_idx, epoch):
        if batch_idx == 0:
            self._reset_metrics()

        x = move_to_device(batch, self.device)['gt']
        with torch.no_grad():
            x_hat, q = self.models['net'](x, mode='full', sample_method='mode')
            x_hat = torch.clamp(x_hat, -1.0, 1.0)

            metrics = self._calc_metrics({'preds': x_hat, 'targets': x})
            metrics['l_kl'] = (q.kl().mean() if hasattr(q, 'kl') else torch.zeros((), device=self.device))
            named_images = {'gt': x, 'recon': x_hat}

        return metrics, named_images
