import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCEDispersiveLoss(nn.Module):

    def __init__(self, tau=0.1, normalize=True, eps=1e-8):
        super().__init__(self)
        self.tau = tau
        self.normalize = normalize
        self.eps = eps

    def forward(self, z):
        """
        InfoNCE-style uniformity term using L2 distances:

            L = log E_{i != j} [ exp( - ||z_i - z_j||^2 / tau ) ]

        Minimizing this pushes pairwise distances ||z_i - z_j|| larger.
        """

        if z.ndim > 2:
            z = z.view(z.size(0), -1)  # [B, D]

        B, D = z.shape
        if B <= 1:
            # no pairwise structure with a single sample
            return z.new_zeros(())

        if self.normalize:
            z = F.normalize(z, dim=1, eps=self.eps)

        # Pairwise squared L2 distances: [B, B]
        diff = z.unsqueeze(1) - z.unsqueeze(0)  # [B, B, D]
        dist2 = (diff ** 2).sum(dim=-1)                     # [B, B]

        # Exclude i == j
        mask = ~torch.eye(B, device=z.device, dtype=torch.bool)
        dist2_off = dist2[mask]  # all i != j, shape [B*(B-1)]

        # InfoNCE-style uniformity over negative squared L2 distances
        scaled = -dist2_off / self.tau
        loss = torch.log(torch.exp(scaled).mean() + self.eps)
        return loss