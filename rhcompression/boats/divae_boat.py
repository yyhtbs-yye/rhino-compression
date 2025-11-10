import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcompression.boats.sdvae_boat import SDVAEBoat

class DIVAEBoat(SDVAEBoat):

    def __init__(self, config=None):
        super().__init__(config=config or {})
        cfg = config or {}
        hps = cfg.get("boat", {}).get("hyperparameters", {})

        # InfoNCE-L2 hyperparameters
        self.disp_tau = float(hps.get("disp_tau", 0.1))
        self.disp_normalize = bool(hps.get("disp_normalize", True))
        self.disp_eps = float(hps.get("disp_eps", 1e-8))

        # Weight of dispersive regularizer
        self.w_disp = float(hps.get("w_disp", 0.0))

    def _dispersive_loss(self, z_samples: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE-style uniformity term using L2 distances:

            L = log E_{i != j} [ exp( - ||z_i - z_j||^2 / tau ) ]

        Minimizing this pushes pairwise distances ||z_i - z_j|| larger.
        """

        if z_samples.ndim > 2:
            z_samples = z_samples.view(z_samples.size(0), -1)  # [B, D]

        B, D = z_samples.shape
        if B <= 1:
            # no pairwise structure with a single sample
            return z_samples.new_zeros(())

        if self.disp_normalize:
            z_samples = F.normalize(z_samples, dim=1, eps=self.disp_eps)

        # Pairwise squared L2 distances: [B, B]
        diff = z_samples.unsqueeze(1) - z_samples.unsqueeze(0)  # [B, B, D]
        dist2 = (diff ** 2).sum(dim=-1)                     # [B, B]

        # Exclude i == j
        mask = ~torch.eye(B, device=z_samples.device, dtype=torch.bool)
        dist2_off = dist2[mask]  # all i != j, shape [B*(B-1)]

        # InfoNCE-style uniformity over negative squared L2 distances
        scaled = -dist2_off / self.disp_tau
        loss = torch.log(torch.exp(scaled).mean() + self.disp_eps)
        return loss

    # ---------- G step with added dispersive term ----------
    def g_step_calc_losses(self, batch):

        losses = super().g_step_calc_losses(batch)

        z_sample = batch['z_sample']

        # Dispersive latent loss (InfoNCE-L2)
        l_disp = self._dispersive_loss(z_sample)

        g_loss = losses.pop('g_loss')

        # Total generator loss
        g_loss = (
            g_loss
            + l_disp * self.w_disp
        )

        return {
            "g_loss": g_loss,
            **losses,
            "l_disp": l_disp.detach(),
        }
