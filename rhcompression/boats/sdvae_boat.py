import torch
from rhtrain.utils.ddp_utils import move_to_device
from rhadversarial.boats.base_gan_boat import BaseGANBoat
EPS = 1e-9

class SDVAEBoat(BaseGANBoat): # Stable Diffusion Variational Autoencoder Boat
    def __init__(self, config=None):
        super().__init__(config=config or {})
        hps  = config['boat'].get('hyperparameters', {})
        self.beta_kl            = float(hps.get('beta_kl', 1e-6))

    # ---------- Inference ----------
    def predict(self, x):
        output = self.models['net'](x, mode='full')
        return torch.clamp(output['x_mean'], -1.0, 1.0), torch.clamp(output['x_sample'], -1.0, 1.0)

    # -------- D step wiring (real vs recon) --------
    def d_step_calc_losses(self, batch):

        critic_weight = self.loss_weight_schedulers['critic'](self.global_step())

        if critic_weight > EPS:

            x = batch['gt']

            with torch.no_grad():
                output = self.models['net'](x, mode='full') 

            d_real = self.models['critic'](x)
            d_fake = self.models['critic'](output['x_mean'].detach())

            train_out = {'real': d_real, 'fake': d_fake, **batch}

            m0 = d_real.mean().detach() - d_fake.mean().detach()

            d_loss = self.losses['critic'](train_out)

            w_adv = critic_weight * d_loss

            return {
                'd_loss': d_loss * w_adv,
                'd_loss_raw': d_loss.detach(),
                'm0_patch': m0.detach(),

            }
        else:
            return {'d_loss': torch.zeros((), device=self.device),
                    'd_loss_raw': torch.zeros((), device=self.device),
                    'm0_patch': torch.zeros((), device=self.device)}

    # ---------- Train ----------
    def g_step_calc_losses(self, batch):

        x = batch['gt']
        
        out = self.models['net'](x, mode='full')

        self.out = out

        results = {}

        results['w_pixel'] = self.loss_weight_schedulers['pixel_loss'](self.global_step())
        results['l_pixel'] = self.losses['pixel_loss'](out['x_sample'], x)

        results['w_lpips'] = self.loss_weight_schedulers['lpips_loss'](self.global_step())
        results['l_lpips'] = self.losses['lpips_loss'](out['x_sample'], x)

        # Adversarial Loss
        results['w_adv'] = self.loss_weight_schedulers['critic'](self.global_step())
        if results['w_adv'] > EPS:
            g_real = self.models['critic'](out['x_sample'])
            results['l_adv'] = self.losses['critic']({'real': g_real, 'fake': None, **batch})
        else:
            results['l_adv'] = torch.zeros((), device=self.device)

        results['l_kl'] = out['q'].kl().mean()

        results['g_loss'] = (
            results['w_pixel'] * results['l_pixel'] 
            + results['w_lpips'] * results['l_lpips'] 
            + results['w_adv'] * results['l_adv'] 
            + self.beta_kl * results['l_kl']
        )

        return {k: v if k == 'g_loss' else torch.tensor(v).detach() for k, v in results.items()}

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
