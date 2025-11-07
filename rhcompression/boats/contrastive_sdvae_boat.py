import torch
import torch.nn as nn
import torch.nn.functional as F

from rhtrain.utils.ddp_utils import move_to_device
from rhcompression.boats.dispersive_sdvae_boat import SDVAEDispersiveBoat

class ContrastiveLoss(nn.Module):
    """
    Full contrastive loss (alignment + uniformity).

    variants:
        - 'infonce'    : SimCLR-style NT-Xent loss.
        - 'hinge'      : Classic contrastive loss with margin (pos + neg).
        - 'covariance' : Barlow-Twins style cross-correlation loss.
    """

    def __init__(
        self,
        variant: str = "infonce",
        tau: float = 0.1,
        margin: float = 1.0,
        bt_lambda: float = 1e-3,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        variant = variant.lower()
        assert variant in {"infonce", "hinge", "covariance"}
        self.variant = variant
        self.tau = tau
        self.margin = margin
        self.bt_lambda = bt_lambda  # off-diagonal weight for covariance variant
        self.normalize = normalize
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: tensor [B, D] or [B, C, H, W]  (view 1)
            z2: tensor [B, D] or [B, C, H, W]  (view 2)

        Returns:
            Scalar loss tensor.
        """
        assert z1.shape[0] == z2.shape[0], "Batch sizes must match"

        if z1.ndim > 2:
            z1 = z1.view(z1.size(0), -1)
            z2 = z2.view(z2.size(0), -1)
        B, D = z1.shape

        if B <= 1:
            return z1.new_zeros(())

        if self.normalize and self.variant in {"infonce", "hinge"}:
            z1 = F.normalize(z1, dim=1, eps=self.eps)
            z2 = F.normalize(z2, dim=1, eps=self.eps)

        if self.variant == "infonce":
            # SimCLR NT-Xent over 2B embeddings
            z = torch.cat([z1, z2], dim=0)       # [2B, D]
            sim = (z @ z.t()) / self.tau         # [2B, 2B]

            # mask out self-similarities
            mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
            sim = sim.masked_fill(mask, float("-inf"))

            # positive index: i <-> i+B (mod 2B)
            pos_idx = (torch.arange(2 * B, device=z.device) + B) % (2 * B)

            log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
            loss = -log_prob[torch.arange(2 * B, device=z.device), pos_idx].mean()
            return loss

        if self.variant == "hinge":
            # Positive term: pull matched pairs together
            diff_pos = z1 - z2
            d_pos = diff_pos.norm(dim=-1)
            loss_pos = (d_pos ** 2).mean()

            # Negative term: push unmatched pairs apart
            diff = z1.unsqueeze(1) - z2.unsqueeze(0)  # [B, B, D]
            d = diff.norm(dim=-1)                     # [B, B]
            mask = ~torch.eye(B, device=z1.device, dtype=torch.bool)
            d_neg = d[mask]
            loss_neg = F.relu(self.margin - d_neg).mean()

            return loss_pos + loss_neg

        # 'covariance' / Barlow Twins variant:
        #   on-diagonal ~ 1 (alignment), off-diagonal ~ 0 (decorrelation)
        z1_c = z1 - z1.mean(dim=0, keepdim=True)
        z2_c = z2 - z2.mean(dim=0, keepdim=True)
        z1_n = z1_c / (z1_c.std(dim=0, keepdim=True) + self.eps)
        z2_n = z2_c / (z2_c.std(dim=0, keepdim=True) + self.eps)

        c = (z1_n.t() @ z2_n) / (B - 1.0)  # [D, D]

        on_diag = torch.diagonal(c)
        loss_on = ((1.0 - on_diag) ** 2).mean()

        eye = torch.eye(D, device=z1.device, dtype=torch.bool)
        off_diag = c[~eye]
        loss_off = (off_diag ** 2).mean()

        loss = loss_on + self.bt_lambda * loss_off
        return loss


class SDVAEContrastiveBoat(SDVAEDispersiveBoat):
    """
    SDVAE + Contrastive latent regularizer.

    New hyperparameters (under boat.hyperparameters):
        ctra_tau        : float, default 0.1
            Temperature in the InfoNCE-style uniformity term.
        ctra_normalize  : bool, default True
            If True, L2-normalize z before computing pairwise distances.
        ctra_use_latent : str, in {"z_feat", "posterior"}, default "z_feat"
            Which latent to regularize:
                "z_feat"     -> second return of net: x_hat, z_feat, q
                "posterior"  -> q.rsample() (or q.mean if no rsample)
    Optional schedule (in boat):
        ctra_fadein: any schedule component; same API as lpips_fadein/adv_fadein.
    """

    def __init__(self, config=None):
        super().__init__(config=config or {})
        cfg = config or {}
        hps = cfg.get("boat", {}).get("hyperparameters", {})

        self.dist_reg_loss = ContrastiveLoss(
            variant=hps.get("ctra_variant", 'infonce'),
            tau=float(hps.get("ctra_tau", 0.1)),
            normalize=bool(hps.get("ctra_normalize", True)),
        )

        self.weight_dist_reg = float(hps.get("weight_dist_reg", 0.0))
