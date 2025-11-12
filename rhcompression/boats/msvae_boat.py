import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcompression.boats.sdvae_boat import SDVAEBoat

EPS = 1e-6

# MeanStay VAE (Mainstay VAE/AnchorVAE)
class MSVAEBoat(SDVAEBoat):
    def __init__(self, config=None):
        super().__init__(config=config or {})

        cfg = config or {}
        hps = cfg.get("boat", {}).get("hyperparameters", {})
        self.weight_adv_patch = float(hps.get('weight_adv_patch', 0.8))

    def d_step_calc_losses(self, batch):

        critic_weight = self.weight_schedulers['critic'](self.global_step())

        if critic_weight > EPS:

            x = batch['gt']

            with torch.no_grad():
                out = self.models['net'](x, mode='full') 

            d_real_patch = self.models['patch_critic'](x)
            d_fake_patch = self.models['patch_critic'](out['x_sample_sg'].detach())
            d_loss_patch = self.losses['critic'](
                {'real': d_real_patch, 'fake': d_fake_patch, **batch}
            )

            d_real_image = self.models['image_critic'](x)
            d_fake_image = self.models['image_critic'](out['x_sample_sg'].detach())
            d_loss_image = self.losses['critic'](
                {'real': d_real_image, 'fake': d_fake_image, **batch}
            )

            d_loss = self.weight_adv_patch * d_loss_patch + (1 - self.weight_adv_patch) * d_loss_image

            return {'d_loss': critic_weight * d_loss}
        else:
            return {'d_loss': torch.zeros((), device=self.device)}

    # ---------- Train ---------- 
    def g_step_calc_losses(self, batch):

        x = batch['gt']

        out = self.models['net'](x, mode='full')

        results = {}
        
        # PixelLoss only affects KL on Mean
        results['w_pixel'] = self.weight_schedulers['pixel_loss'](self.global_step())
        results['l_pixel'] = self.losses['pixel_loss'](out['x_mean'], x)
        
        # # LPIPS does not affect KL at all
        # results['w_lpips'] = self.weight_schedulers['lpips_loss'](self.global_step())
        # results['l_lpips'] = self.losses['lpips_loss'](out['x_mean_sg'], x).sum() + self.losses['lpips_loss'](out['x_sample_sg'], x).sum()

        # LPFLoss affects KL on Mean and Variance
        results['w_lpf'] = self.weight_schedulers['lpf_loss'](self.global_step())
        results['l_lpf'] = self.losses['lpf_loss'](out['x_sample'], x, 0.75 * torch.ones(x.size(0), device=x.device, dtype=x.dtype))

        results['w_adv'] = self.weight_schedulers['critic'](self.global_step())

        # Adversarial Loss
        if results['w_adv'] > EPS:
            g_real_patch = self.models['patch_critic'](out['x_sample_sg'])
            l_adv_patch = self.losses['critic']({'real': g_real_patch, 'fake': None, **batch})
            g_real_image = self.models['image_critic'](out['x_sample_sg'])
            l_adv_image = self.losses['critic']({'real': g_real_image, 'fake': None, **batch})
            results['l_adv'] = self.weight_adv_patch * l_adv_patch + (1 - self.weight_adv_patch) * l_adv_image
        else:
            results['l_adv'] = torch.zeros((), device=self.device)

        results['l_kl'] = out['q'].kl().mean()

        results['g_loss'] = (
            results['w_pixel'] * results['l_pixel'] 
            # + results['w_lpips'] * results['l_lpips'] 
            + results['w_lpf'] * results['l_lpf'] 
            + results['w_adv'] * results['l_adv'] 
            + self.beta_kl * results['l_kl']
        )

        return {k: v if k == 'g_loss' else torch.tensor(v).detach() for k, v in results.items()}

