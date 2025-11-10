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
        self.max_weight_adv     = float(hps.get('max_weight_adv', 1.0))
        self.start_adv          = int(hps.get('start_adv', 20_000))

        # Annealers are optional; default to constant schedules
        self.lpips_fadein = build_module(config['boat']['lpips_fadein']) if 'lpips_fadein' in config['boat'] else (lambda *_: 0.0)
        self.adv_fadein = build_module(config['boat']['adv_fadein']) if 'adv_fadein' in config['boat'] else (lambda *_: 0.0)

    # ---------- Inference ----------
    def predict(self, x):
        output = self.models['net'](x, mode='full')
        return torch.clamp(output['x_mean'], -1.0, 1.0), torch.clamp(output['x_sample'], -1.0, 1.0)

    # -------- D step wiring (real vs recon) --------
    def d_step_calc_losses(self, batch):

        if self.global_step() >= self.start_adv:

            batch = move_to_device(batch, self.device)
            x = batch['gt']

            with torch.no_grad():
                output = self.models['net'](x, mode='full') 

            d_real = self.models['critic'](x)
            d_fake = self.models['critic'](output['x_mean'].detach())

            train_out = {'real': d_real, 'fake': d_fake, **batch}

            d_loss = self.losses['critic'](train_out)

            w_adv = self.max_weight_adv * float(self.adv_fadein(self.global_step()))

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

        x = batch['gt']
        
        output = self.models['net'](x, mode='full')

        l_image = self.losses['pixel_loss'](output['x_sample'], x)

        w_adv = self.max_weight_adv * float(self.adv_fadein(self.global_step()))
        if self.global_step() >= self.start_adv:
            d_fake_for_g = self.models['critic'](output['x_mean'])
            train_out = {'real': d_fake_for_g, 'fake': None, **batch}
            l_adv = self.losses['critic'](train_out)
        else:
            l_adv = torch.zeros((), device=self.device)

        w_lpips = self.max_weight_lpips * float(self.lpips_fadein(self.global_step()))
        l_lpips = (self.losses['lpips_loss'](output['x_mean'], x).mean()
                   if ('lpips_loss' in self.losses and w_lpips > 1e-6)
                   else torch.zeros((), device=self.device))

        l_kl = output['q'].kl().mean()

        g_loss = (
            self.lambda_image * l_image 
            + w_adv * l_adv 
            + w_lpips * l_lpips 
            + self.beta_kl * l_kl
        )

        batch['z_sample'] = output['z_sample']
        batch['x_sample'] = output['x_sample']
        batch['z_mean'] = output['z_mean']
        batch['x_mean'] = output['x_mean']
        
        return {
            'g_loss': g_loss,
            'l_image': l_image.detach(),
            'w_adv': torch.tensor(w_adv),
            'l_adv': l_adv.detach(),
            'w_lpips': torch.tensor(w_lpips),
            'l_lpips': l_lpips.detach(),
            'l_kl': l_kl.detach(),
        }

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
