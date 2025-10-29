import torch
from rhtrain.utils.ddp_utils import move_to_device
from rhadversarial.boats.base_gan_boat import BaseGANBoat
from rhcore.utils.build_components import build_module

class SDVAEBoat(BaseGANBoat): # Stable Diffusion Variational Autoencoder Boat
    def __init__(self, config=None):
        super().__init__(config=config or {})
        hps  = config['boat'].get('hyperparameters', {})

        self.lambda_image       = float(hps.get('lambda_image', 1.0))
        self.beta_kl            = float(hps.get('beta_kl', 1e-6))
        self.max_weight_lpips   = float(hps.get('max_weight_lpips', 1.0))
        self.max_weight_adv         = float(hps.get('max_weight_adv', 1.0))
        self.use_mode_eval      = bool(hps.get('use_mean_at_eval', True))
        self.start_adv          = int(hps.get('start_adv', 20_000))
        # Annealers are optional; default to constant schedules
        self.lpips_fadein = build_module(config['boat']['lpips_fadein']) if 'lpips_fadein' in config['boat'] else (lambda *_: 0.0)
        self.adv_fadein = build_module(config['boat']['adv_fadein']) if 'adv_fadein' in config['boat'] else (lambda *_: 0.0)

    # ---------- Inference ----------
    def predict(self, x):
        sm = 'mode' if self.use_mode_eval else 'random'
        x_hat, _ = self.models['net'](x, mode='full', sample_method=sm)
        return torch.clamp(x_hat, -1.0, 1.0)

    # -------- D step wiring (real vs recon) --------
    def d_step_calc_losses(self, batch):

        if self.global_step() >= self.start_adv:

            batch = move_to_device(batch, self.device)
            x = batch['gt']

            with torch.no_grad():
                x_hat, _ = self.models['net'](x, mode='full', sample_method='mode')

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
        
        x_hat, q = self.models['net'](x, mode='full', sample_method='random')

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

        l_kl = (q.kl().mean() if hasattr(q, 'kl') else torch.zeros((), device=self.device))

        g_loss = self.lambda_image * l_img + w_adv * l_adv + w_lpips * l_lpips + self.beta_kl * l_kl

        return {
            'g_loss': g_loss,
            'l_image': l_img.detach(),
            'l_adv': l_adv.detach(),
            'l_lpips': l_lpips.detach(),
            'l_kl': l_kl.detach(),
            'w_lpips': torch.tensor(w_lpips),
            'w_adv': torch.tensor(w_adv),
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
