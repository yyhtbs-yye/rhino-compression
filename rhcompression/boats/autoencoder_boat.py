import math
import torch
import torch.nn.functional as F

from rhtrain.utils.ddp_utils import move_to_device
from rhcore.boats.base_boat import BaseBoat
from rhcore.utils.build_components import build_module

class AutoencoderBoat(BaseBoat):

    def __init__(self, config={}):
        super().__init__(config=config)
        hps  = config['boat'].get('hyperparameters', {})

        self.lambda_image  = float(hps.get('lambda_image', 1.0))
        self.max_w_lpips  = float(hps.get('max_w_lpips', 1.0))

        self.lpips_fadein = build_module(config['boat']['lpips_fadein'])

    def predict(self, x):
        z = self.models['net'](x, mode='encode')
        x_hat = torch.clamp(self.models['net'](z, mode='decode'), min=-1.0, max=1.0)

        return x_hat

    def training_calc_losses(self, batch):
        batch = move_to_device(batch, self.device)
        x = batch['gt']

        z = self.models['net'](x, mode='encode')   
        x_hat = self.models['net'](z, mode='decode')

        # Reconstruction
        l_image = self.losses['pixel_loss'](x_hat, x)

        # Perceptual 
        l_lpips = torch.zeros((), device=self.device)

        w_lpips = self.max_w_lpips * (1.0 - self.lpips_fadein(self.global_step()))

        if w_lpips > 1e-6 and 'lpips_loss' in self.losses:
            lp = self.losses['lpips_loss'](x_hat, x)
            l_lpips = (lp.mean() if lp.ndim > 0 else lp)

        total_loss = l_image * self.lambda_image + l_lpips * w_lpips

        # Return losses + metrics (your logger can pick these up)
        return {
            'total_loss': total_loss,
            'l_image': l_image.detach(),
            'l_lpips': l_lpips.detach(),
            'w_lpips': torch.tensor(w_lpips),
        }

    # -------- Validation: reconstruct inputs --------
    def validation_step(self, batch, batch_idx, epoch):
        if batch_idx == 0:
            self._reset_metrics()

        batch = move_to_device(batch, self.device)
        x = batch['gt']

        with torch.no_grad():
            x_hat = self.predict(x)
            valid_output = {'preds': x_hat, 'targets': x}
            metrics = self._calc_metrics(valid_output)
            named_images = {'gt': x, 'recon': x_hat}

        return metrics, named_images
