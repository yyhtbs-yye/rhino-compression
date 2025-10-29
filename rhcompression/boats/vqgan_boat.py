import torch
from rhtrain.utils.ddp_utils import move_to_device, get_raw_module
from rhadversarial.boats.base_gan_boat import BaseGANBoat
from rhcore.utils.build_components import build_module

class VQGANBoat(BaseGANBoat): # VQVAE - GAN Boat
    def __init__(self, config=None):
        super().__init__(config=config or {})
        hps  = config['boat'].get('hyperparameters', {})

        self.lambda_image       = float(hps.get('lambda_image', 1.0))
        self.max_weight_lpips   = float(hps.get('max_weight_lpips', 1.0))
        self.max_weight_adv     = float(hps.get('max_weight_adv', 1.0))
        self.use_mode_eval      = bool(hps.get('use_mean_at_eval', True))
        self.start_adv          = int(hps.get('start_adv', 20_000))
        
        # Annealers are optional; default to constant schedules
        self.lpips_fadein = build_module(config['boat']['lpips_fadein']) if 'lpips_fadein' in config['boat'] else (lambda *_: 0.0)
        self.adv_fadein = build_module(config['boat']['adv_fadein']) if 'adv_fadein' in config['boat'] else (lambda *_: 0.0)

    # ---------- Inference ----------
    def predict(self, x):
        x_hat, _, _ = self.models['net'](x, mode='full')
        return torch.clamp(x_hat, -1.0, 1.0)

    # -------- D step wiring (real vs recon) --------
    def d_step_calc_losses(self, batch):

        if self.global_step() >= self.start_adv:

            batch = move_to_device(batch, self.device)
            x = batch['gt']

            with torch.no_grad():
                x_hat, _, _ = self.models['net'](x, mode='full')

            d_real = self.models['critic'](x)
            d_fake = self.models['critic'](x_hat.detach())

            train_out = {'real': d_real, 'fake': d_fake, **batch}

            d_loss = self.losses['critic'](train_out)

            w_adv = self.max_weight_adv * (1.0 - float(self.adv_fadein(self.global_step())))

            return {
                'd_loss': d_loss * w_adv,
                'd_real': d_real.mean().detach(),
                'd_fake': d_fake.mean().detach(),
            }
        else:
            return {'d_loss': torch.zeros((), device=self.device),
                    'd_real': torch.zeros((), device=self.device),
                    'd_fake': torch.zeros((), device=self.device)}

    # ---------- Train ----------
    def g_step_calc_losses(self, batch):

        batch = move_to_device(batch, self.device)
        
        x = batch['gt']
        
        x_hat, qloss, indices = self.models['net'](x, mode='full')

        l_img = self.losses['pixel_loss'](x_hat, x)

        if self.global_step() >= self.start_adv:
            d_fake_for_g = self.models['critic'](x_hat)
            train_out = {'real': d_fake_for_g, 'fake': None, **batch}
            l_adv = self.losses['critic'](train_out)
        else:
            l_adv = torch.zeros((), device=self.device)

        w_lpips = self.max_weight_lpips * float(self.lpips_fadein(self.global_step()))
        w_adv = self.max_weight_adv * float(self.adv_fadein(self.global_step()))

        l_lpips = (self.losses['lpips_loss'](x_hat, x).mean()
                   if ('lpips_loss' in self.losses and w_lpips > 1e-6)
                   else torch.zeros((), device=self.device))

        unique_indices = torch.unique(indices)
        dict_usage = unique_indices / get_raw_module(self.models['net']).vocab_size

        g_loss = self.lambda_image * l_img + w_adv * l_adv + w_lpips * l_lpips

        return {
            'g_loss': g_loss,
            'l_image': l_img.detach(),
            'l_adv': l_adv.detach(),
            'l_lpips': l_lpips.detach(),
            'w_lpips': torch.tensor(w_lpips),
            'w_adv': torch.tensor(w_adv),
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
