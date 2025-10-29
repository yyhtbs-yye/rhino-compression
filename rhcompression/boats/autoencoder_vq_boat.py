import torch
from rhtrain.utils.ddp_utils import move_to_device, get_raw_module
from rhcore.boats.base_boat import BaseBoat
from rhcore.utils.build_components import build_module

class AutoencoderVQBoat(BaseBoat):
    def __init__(self, config=None):
        super().__init__(config=config or {})
        hps  = config['boat'].get('hyperparameters', {})

        self.lambda_image   = float(hps.get('lambda_image', 1.0))
        self.max_weight_lpips    = float(hps.get('max_weight_lpips', 1.0))

        # Annealers are optional; default to constant schedules
        self.lpips_fadein = build_module(config['boat']['lpips_fadein']) if 'lpips_fadein' in config['boat'] else (lambda *_: 0.0)

    # ---------- Inference ----------
    def predict(self, x):
        
        x_hat, _, _ = self.models['net'](x, mode='full')
        return torch.clamp(x_hat, -1.0, 1.0)

    # ---------- Train ----------
    def training_calc_losses(self, batch):
        x = move_to_device(batch, self.device)['gt']
        x_hat, qloss, indices = self.models['net'](x, mode='full')

        l_img = self.losses['pixel_loss'](x_hat, x)

        w_lpips = self.max_weight_lpips * float(self.lpips_fadein(self.global_step()))

        l_lpips = (self.losses['lpips_loss'](x_hat, x).mean()
                   if ('lpips_loss' in self.losses and w_lpips > 1e-6)
                   else torch.zeros((), device=self.device))
        
        unique_indices = torch.unique(indices)
        dict_usage = unique_indices / get_raw_module(self.models['net']).vocab_size

        total = self.lambda_image * l_img + w_lpips * l_lpips

        return {
            'total_loss': total,
            'l_image': l_img.detach(),
            'l_lpips': l_lpips.detach(),
            'w_lpips': torch.tensor(w_lpips),
            'l_vq': qloss,
            'dict_usage': dict_usage,
        }

    # ---------- Validation ----------
    def validation_step(self, batch, batch_idx, epoch):
        if batch_idx == 0:
            self._reset_metrics()

        x = move_to_device(batch, self.device)['gt']
        with torch.no_grad():
            x_hat, q_loss, _ = self.models['net'](x, mode='full')
            x_hat = torch.clamp(x_hat, -1.0, 1.0)

            metrics = self._calc_metrics({'preds': x_hat, 'targets': x})
            metrics['q_loss'] = q_loss.detach()
            named_images = {'gt': x, 'recon': x_hat}

        return metrics, named_images
