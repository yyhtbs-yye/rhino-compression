import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcompression.nn.losses.utils.kernels import rbf_kernel

class RBFKernelMMDLoss(nn.Module):

    def __init__(self, sigma=1.0):
        super().__init__(self)
        self.sigma = sigma

    def _mmd_loss(self, z):
        """
        MMD^2(q(z), p(z)) with p(z) = N(0, I).
        z: (B, D) samples from q(z|x)
        """
        z = z.flatten(1)
        prior_samples = torch.randn_like(z)

        k_zz = rbf_kernel(z, z, self.sigma)
        k_pp = rbf_kernel(prior_samples, prior_samples, self.sigma)
        k_zp = rbf_kernel(z, prior_samples, self.sigma)

        mmd2 = k_zz.mean() + k_pp.mean() - 2.0 * k_zp.mean()

        return mmd2
