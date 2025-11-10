import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcompression.boats.sdvae_boat import SDVAEBoat
    
def rbf_kernel(x, y, sigma):
    """
    x: (B, D), y: (B, D)
    returns K in R^{B x B}
    """
    x = x.unsqueeze(1)  # (B, 1, D)
    y = y.unsqueeze(0)  # (1, B, D)
    diff = x - y
    dist_sq = (diff ** 2).sum(-1)  # (B, B)
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))

class InfoVAEBoat(SDVAEBoat):

    def __init__(self, config=None):
        super().__init__(config=config or {})
        hps = (config or {}).get("boat", {}).get("hyperparameters", {})

        # InfoVAE MMD hyperparameters
        self.mmd_weight = float(hps.get("beta_kl", 1.0)) * 100
        self.mmd_sigma = float(hps.get("mmd_sigma", 1.0))

    def _mmd_loss(self, z_samples):
        """
        MMD^2(q(z), p(z)) with p(z) = N(0, I).
        z_samples: (B, D) samples from q(z|x)
        """
        prior_samples = torch.randn_like(z_samples)

        k_zz = rbf_kernel(z_samples, z_samples, self.mmd_sigma)
        k_pp = rbf_kernel(prior_samples, prior_samples, self.mmd_sigma)
        k_zp = rbf_kernel(z_samples, prior_samples, self.mmd_sigma)

        mmd2 = k_zz.mean() + k_pp.mean() - 2.0 * k_zp.mean()

        return mmd2

    # ---------- Train ----------
    # d_step_calc_losses is inherited from SDVAEBoat
    def g_step_calc_losses(self, batch):

        losses = super().g_step_calc_losses(batch)

        z_sample = batch['z_sample']

        # InfoVAE MMD term on z ~ q(z|x)
        z_sample = z_sample.view(z_sample.size(0), -1)
        l_mmd = self._mmd_loss(z_sample)

        g_loss = losses.pop('g_loss')

        g_loss = (
            g_loss 
            + self.mmd_weight * l_mmd
        )

        return {
            "g_loss": g_loss,
            **losses,
            "l_mmd": l_mmd.detach(),
        }
