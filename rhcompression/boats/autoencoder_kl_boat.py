import torch
from rhtrain.utils.ddp_utils import move_to_device
from rhcore.boats.base_boat import BaseBoat
from rhcore.utils.build_components import build_module

class AutoencoderKLBoat(BaseBoat):
    def __init__(self, config=None):
        super().__init__(config=config or {})
        hps  = config['boat'].get('hyperparameters', {})
        self.beta_kl        = float(hps.get('beta_kl', hps.get('lambda_kl', 1e-6)))

        # Annealers are optional; default to constant schedules
        self.lpips_fadein = build_module(config['boat']['lpips_fadein']) if 'lpips_fadein' in config['boat'] else (lambda *_: 0.0)

    # ---------- Inference ----------
    def predict(self, x):
        output = self.models['net'](x, mode='full')
        return torch.clamp(output['x_mean'], -1.0, 1.0), torch.clamp(output['x_sample'], -1.0, 1.0)

    # ---------- Train ----------
    def training_calc_losses(self, batch):
        x = batch['gt']
        out = self.models['net'](x, mode='full', sample_method='random')

        results = {}

        results['w_pixel'] = self.loss_weight_schedulers['pixel_loss'](self.global_step())
        results['l_pixel'] = self.losses['pixel_loss'](out['x_sample'], x)

        results['w_lpips'] = self.loss_weight_schedulers['lpips_loss'](self.global_step())
        results['l_lpips'] = self.losses['lpips_loss'](out['x_sample'], x)

        results['l_kl'] = out['q'].kl().mean()

        results['total_loss'] = (
            results['w_pixel'] * results['l_pixel'] 
            + results['w_lpips'] * results['l_lpips'] 
            + self.beta_kl * results['l_kl']
        )

        return {k: v if k == 'total_loss' else torch.tensor(v).detach() for k, v in results.items()}

    # ---------- Validation ----------
    def validation_step(self, batch, batch_idx, epoch):
        if batch_idx == 0:
            self._reset_metrics()

        x = move_to_device(batch, self.device)['gt']
        with torch.no_grad():
            output = self.models['net'](x, mode='full')
            x_mean = torch.clamp(output['x_mean'], -1.0, 1.0)
            x_sample = torch.clamp(output['x_sample'], -1.0, 1.0)

            metrics = self._calc_metrics({'preds': x_mean, 'targets': x})
            metrics['l_kl'] = output['q'].kl().mean()
            named_images = {'gt': x, 'recon': x_mean, 'rand_preds': x_sample}

        return metrics, named_images
