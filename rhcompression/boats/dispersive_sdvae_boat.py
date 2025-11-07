import torch
import torch.nn as nn
import torch.nn.functional as F

from rhtrain.utils.ddp_utils import move_to_device
from sdvae_boat import SDVAEBoat

class DispersiveLoss(nn.Module):
    """
    Dispersive loss: keeps only the "repulsive" / uniformity part
    of a contrastive objective.

    variants:
        - 'infonce'    : InfoNCE-style uniformity term over a single batch.
        - 'hinge'      : Negative-pair hinge on distances.
        - 'covariance' : Off-diagonal covariance penalty (Barlow-Twins style).
    """

    def __init__(
        self,
        variant: str = "infonce",
        tau: float = 0.1,
        margin: float = 1.0,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        variant = variant.lower()
        assert variant in {"infonce", "hinge", "covariance"}
        self.variant = variant
        self.tau = tau
        self.margin = margin
        self.normalize = normalize
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: tensor of shape [B, D] or [B, C, H, W]

        Returns:
            Scalar loss tensor.
        """
        if z.ndim > 2:
            z = z.view(z.size(0), -1)  # [B, D]
        B, D = z.shape

        if B <= 1:
            return z.new_zeros(())

        if self.normalize and self.variant in {"infonce", "hinge"}:
            z = F.normalize(z, dim=1, eps=self.eps)

        if self.variant == "infonce":
            # InfoNCE-style uniformity: log E_{i!=j} exp(sim_ij / tau)
            sim = (z @ z.t()) / self.tau  # [B, B]
            mask = ~torch.eye(B, device=z.device, dtype=torch.bool)
            sim_off = sim[mask]  # all i != j
            loss = torch.log(torch.exp(sim_off).mean() + self.eps)
            return loss

        if self.variant == "hinge":
            # Negative-only hinge on distances between distinct samples
            diff = z.unsqueeze(1) - z.unsqueeze(0)  # [B, B, D]
            d = diff.norm(dim=-1)                   # [B, B]
            mask = ~torch.eye(B, device=z.device, dtype=torch.bool)
            d_off = d[mask]
            loss = F.relu(self.margin - d_off).mean()
            return loss

        # 'covariance' variant:
        #   decorrelate features across the batch via off-diagonal cov penalty
        z = z - z.mean(dim=0, keepdim=True)
        std = z.std(dim=0, keepdim=True) + self.eps
        z_norm = z / std
        cov = (z_norm.t() @ z_norm) / (B - 1.0)  # [D, D]

        eye = torch.eye(D, device=z.device, dtype=torch.bool)
        off_diag = cov[~eye]
        loss = (off_diag ** 2).mean()
        return loss

class SDVAEDispersiveBoat(SDVAEBoat):
    """
    SDVAE + Dispersive latent regularizer.

    New hyperparameters (under boat.hyperparameters):
        max_weight_dist_reg : float, default 0.0
            Maximum weight for the dispersive loss.
        disp_tau        : float, default 0.1
            Temperature in the InfoNCE-style uniformity term.
        disp_normalize  : bool, default True
            If True, L2-normalize z before computing pairwise distances.
        disp_use_latent : str, in {"z_feat", "posterior"}, default "z_feat"
            Which latent to regularize:
                "z_feat"     -> second return of net: x_hat, z_feat, q
                "posterior"  -> q.rsample() (or q.mean if no rsample)
    Optional schedule (in boat):
        disp_fadein: any schedule component; same API as lpips_fadein/adv_fadein.
    """

    def __init__(self, config=None):
        super().__init__(config=config or {})
        cfg = config or {}
        hps = cfg.get("boat", {}).get("hyperparameters", {})

        self.dist_reg_loss = DispersiveLoss(
            variant=hps.get("disp_variant", 'infonce'),
            tau=float(hps.get("disp_tau", 0.1)),
            normalize=bool(hps.get("disp_normalize", True)),
        )

        self.weight_dist_reg = float(hps.get("weight_dist_reg", 0.0))

    # ---------- G step with added dispersive term ----------
    def g_step_calc_losses(self, batch):

        batch = move_to_device(batch, self.device)
        x = batch["gt"]

        # IMPORTANT: here we keep the second output (z_feat)
        x_hat, z_feat, q = self.models["net"](x, mode="full", sample_method="random")

        # Pixel reconstruction loss
        l_img = self.losses["pixel_loss"](x_hat, x)

        # Adversarial term (unchanged)
        w_adv = self.max_weight_adv * float(self.adv_fadein(self.global_step()))
        if self.global_step() >= self.start_adv:
            d_fake_for_g = self.models["critic"](x_hat)
            train_out = {"real": d_fake_for_g, "fake": None, **batch}
            l_adv = self.losses["critic"](train_out)
        else:
            l_adv = torch.zeros((), device=self.device)

        # LPIPS term (unchanged)
        w_lpips = self.max_weight_lpips * float(self.lpips_fadein(self.global_step()))
        l_lpips = (
            self.losses["lpips_loss"](x_hat, x).mean()
            if ("lpips_loss" in self.losses and w_lpips > 1e-6)
            else torch.zeros((), device=self.device)
        )

        # KL term (unchanged)
        l_kl = q.kl().mean() if hasattr(q, "kl") else torch.zeros((), device=self.device)

        z_mu = q.mode()

        l_dist_reg = self.dist_reg_loss(z_mu)

        # Total generator loss
        g_loss = (
            self.lambda_image * l_img
            + w_adv * l_adv
            + w_lpips * l_lpips
            + self.beta_kl * l_kl
            + self.weight_dist_reg * l_dist_reg
        )

        return {
            "g_loss": g_loss,
            "l_image": l_img.detach(),
            "w_adv": torch.tensor(w_adv),
            "l_adv": l_adv.detach(),
            "w_lpips": torch.tensor(w_lpips),
            "l_lpips": l_lpips.detach(),
            "l_kl": l_kl.detach(),
            "l_dist_reg": l_dist_reg.detach(),
        }
