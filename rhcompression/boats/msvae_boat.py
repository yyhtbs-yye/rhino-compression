import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcompression.boats.sdvae_boat import SDVAEBoat
EPS = 1e-9
class MSVAEBoat(SDVAEBoat): # MeanStay VAE (Mainstay VAE/AnchorVAE)

    def __init__(self, config=None):
        super().__init__(config=config or {})

        cfg = config or {}
        hps = cfg.get("boat", {}).get("hyperparameters", {})
        self.weight_adv_patch = float(hps.get('weight_adv_patch', 0.5))
        self.cut_freq = float(hps.get('cut_freq', 0.33))
        self.loss_debug = False

    def d_step_calc_losses(self, batch):

        critic_weight = self.loss_weight_schedulers['critic'](self.global_step())

        if critic_weight > EPS:

            x = batch['gt']

            with torch.no_grad():
                out = self.models['net'](x, mode='full') 

            d_real_patch = self.models['patch_critic'](x)
            d_fake_patch = self.models['patch_critic'](out['x_mean'].detach())
            d_loss_patch = self.losses['critic'](
                {'real': d_real_patch, 'fake': d_fake_patch, **batch}
            )

            m0_patch = d_real_patch.mean().detach() - d_fake_patch.mean().detach()

            d_real_image = self.models['image_critic'](x)
            d_fake_image = self.models['image_critic'](out['x_sample'].detach())
            d_loss_image = self.losses['critic'](
                {'real': d_real_image, 'fake': d_fake_image, **batch}
            )

            m0_image = d_real_image.mean().detach() - d_fake_image.mean().detach()

            d_loss = self.weight_adv_patch * d_loss_patch + (1 - self.weight_adv_patch) * d_loss_image

            return {'d_loss': critic_weight * d_loss,
                    'd_loss_patch': d_loss_patch.detach(),
                    'd_loss_image': d_loss_image.detach(),
                    'm0_patch': m0_patch.detach(),
                    'm0_image': m0_image.detach(),
                    'd_loss_raw': d_loss.detach()}
        else:
            return {'d_loss': torch.zeros((), device=self.device),
                    'd_loss_patch': torch.zeros((), device=self.device),
                    'd_loss_image': torch.zeros((), device=self.device),
                    'm0_patch': torch.zeros((), device=self.device),
                    'm0_image': torch.zeros((), device=self.device),
                    'd_loss_raw': torch.zeros((), device=self.device)}

    def g_step_calc_losses(self, batch):

        x = batch['gt']

        out = self.models['net'](x, mode='full')

        results = {}
        
        # PixelLoss only affects KL on Mean
        results['w_pixel'] = self.loss_weight_schedulers['pixel_loss'](self.global_step())
        results['l_pixel'] = self.losses['pixel_loss'](out['x_mean'], x)
        
        # LPFLoss affects KL on Mean and Variance
        results['w_lpf'] = self.loss_weight_schedulers['lpf_loss'](self.global_step())
        results['l_lpf'] = self.losses['lpf_loss'](out['x_sample'], x, self.cut_freq * torch.ones(x.size(0), device=x.device, dtype=x.dtype))
        results['w_adv'] = self.loss_weight_schedulers['critic'](self.global_step())

        # Adversarial Loss
        if results['w_adv'] > EPS:
            g_real_patch = self.models['patch_critic'](out['x_mean'])
            l_adv_patch = self.losses['critic']({'real': g_real_patch, 'fake': None, **batch})
            g_real_image = self.models['image_critic'](out['x_sample'])
            l_adv_image = self.losses['critic']({'real': g_real_image, 'fake': None, **batch})
            results['l_adv'] = self.weight_adv_patch * l_adv_patch + (1 - self.weight_adv_patch) * l_adv_image
        else:
            results['l_adv'] = torch.zeros((), device=self.device)

        results['l_kl'] = out['q'].kl().mean()

        results['g_loss'] = (
            results['w_pixel'] * results['l_pixel'] 
            + results['w_lpf'] * results['l_lpf'] 
            + results['w_adv'] * results['l_adv'] 
            + self.beta_kl * results['l_kl']
        )

        return {k: v if k == 'g_loss' else torch.tensor(v).detach() for k, v in results.items()}

