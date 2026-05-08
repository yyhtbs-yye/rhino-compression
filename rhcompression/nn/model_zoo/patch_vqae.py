import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# EMA Vector Quantizer (VQ-VAE w/ EMA codebook)
# -----------------------------
class EMAVectorQuantizer(nn.Module):
    """
    EMA-updated codebook with commitment loss only (no codebook loss).
    Quantizes (N, D) latents to nearest code (K, D), returns indices + straight-through z_q.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        kmeans_subsample: int = 131072,
        use_cosine: bool = False,   # if True, use cosine distance (normalize z and embed)
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.beta = beta
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps

        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.kmeans_subsample = kmeans_subsample
        self.use_cosine = use_cosine

        embed = torch.randn(vocab_size, embed_dim)
        embed = F.normalize(embed, dim=1) if use_cosine else embed
        self.embedding = nn.Parameter(embed, requires_grad=False)  # EMA only

        self.register_buffer("ema_cluster_size", torch.zeros(vocab_size))
        self.register_buffer("ema_weight", torch.zeros(vocab_size, embed_dim))
        self.register_buffer("codebook_initialized", torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def _kmeans_init_from_batch(self, z_flat: torch.Tensor):
        # z_flat: (N, D)
        if z_flat.size(0) > self.kmeans_subsample:
            idx = torch.randperm(z_flat.size(0), device=z_flat.device)[: self.kmeans_subsample]
            x = z_flat[idx]
        else:
            x = z_flat

        if self.use_cosine:
            x = F.normalize(x, dim=1)

        # init centers by random samples
        perm = torch.randperm(x.size(0), device=x.device)
        centers = x[perm[: self.vocab_size]].clone()

        # Lloyd iterations
        for _ in range(self.kmeans_iters):
            # distances (M, K)
            if self.use_cosine:
                # cosine distance = 1 - dot for normalized vectors
                dist = 1.0 - (x @ centers.t())
            else:
                x2 = (x ** 2).sum(dim=1, keepdim=True)
                c2 = (centers ** 2).sum(dim=1).unsqueeze(0)
                dist = x2 + c2 - 2.0 * (x @ centers.t())

            labels = dist.argmin(dim=1)
            for k in range(self.vocab_size):
                mask = labels == k
                if mask.any():
                    centers[k] = x[mask].mean(dim=0)
                else:
                    j = torch.randint(0, x.size(0), (1,), device=x.device)
                    centers[k] = x[j]

            if self.use_cosine:
                centers = F.normalize(centers, dim=1)

        self.embedding.data.copy_(centers)
        self.ema_weight.data.copy_(centers)
        self.ema_cluster_size.data.fill_(1.0)
        self.codebook_initialized.fill_(1)

    @torch.no_grad()
    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor):
        # z_flat: (N, D), indices: (N,)
        N, D = z_flat.shape
        K = self.vocab_size

        if self.use_cosine:
            z_flat = F.normalize(z_flat, dim=1)

        enc = torch.zeros(N, K, device=z_flat.device)
        enc.scatter_(1, indices.view(-1, 1), 1)

        cluster_size = enc.sum(0)  # (K,)
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size * (1.0 - self.ema_decay))

        dw = enc.t() @ z_flat  # (K, D)
        self.ema_weight.mul_(self.ema_decay).add_(dw * (1.0 - self.ema_decay))

        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.ema_eps) / (n + K * self.ema_eps) * n
        embed = self.ema_weight / cluster_size.unsqueeze(1)

        if self.use_cosine:
            embed = F.normalize(embed, dim=1)

        self.embedding.data.copy_(embed)

    def quantize(self, z: torch.Tensor):
        """
        z: (N, D) encoder latents
        returns:
          z_q: (N, D) quantized (straight-through)
          indices: (N,)
          qloss: scalar commitment loss
        """
        if self.kmeans_init and self.codebook_initialized.item() == 0:
            with torch.no_grad():
                self._kmeans_init_from_batch(z)

        if self.use_cosine:
            z_in = F.normalize(z, dim=1)
            e = self.embedding.data  # already normalized
            # dist = 1 - dot
            dist = 1.0 - (z_in @ e.t())  # (N, K)
        else:
            z2 = (z ** 2).sum(dim=1, keepdim=True)             # (N,1)
            e2 = (self.embedding.data ** 2).sum(dim=1).unsqueeze(0)  # (1,K)
            dist = z2 + e2 - 2.0 * (z @ self.embedding.data.t())     # (N,K)

        indices = dist.argmin(dim=1)             # (N,)
        z_q = self.embedding[indices]            # (N,D)

        if self.training:
            with torch.no_grad():
                self._ema_update(z, indices)

        # commitment loss + straight-through estimator
        qloss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()

        return z_q, indices, qloss

    @torch.no_grad()
    def indices_to_latents(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (...,) -> latents: (..., D)
        return self.embedding[indices]


# -----------------------------
# Tiny Patch Encoder/Decoder (MLP)
# -----------------------------
class PatchMLPEncoder(nn.Module):
    def __init__(self, patch_dim: int, embed_dim: int, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        layers = []
        d_in = patch_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d_in, hidden_dim), nn.GELU()]
            d_in = hidden_dim
        layers += [nn.Linear(d_in, embed_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # (N, patch_dim) -> (N, embed_dim)
        return self.net(x_flat)


class PatchMLPDecoder(nn.Module):
    def __init__(self, embed_dim: int, patch_dim: int, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        layers = []
        d_in = embed_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d_in, hidden_dim), nn.GELU()]
            d_in = hidden_dim
        layers += [nn.Linear(d_in, patch_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # (N, embed_dim) -> (N, patch_dim)
        return self.net(z)


# -----------------------------
# Patch-VQ Autoencoder: patch -> latent -> code -> latent -> patch recon
# -----------------------------
class PatchVQAutoencoder(nn.Module):
    """
    Works on full images by patchifying internally (non-overlapping patches),
    but the bottleneck is *per patch* (tiny 4x4 / 8x8).
    """
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 8,          # 4 or 8
        embed_dim: int = 128,
        vocab_size: int = 1024,
        enc_hidden: int = 256,
        dec_hidden: int = 256,
        mlp_depth: int = 2,
        vq_beta: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        kmeans_init: bool = True,
        use_cosine: bool = False,
    ):
        super().__init__()
        assert patch_size in (4, 8, 16, 32), "Patch sizes like 4 or 8 are typical."
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = None  # created dynamically once we know H,W at runtime

        self.encoder = PatchMLPEncoder(self.patch_dim, embed_dim, hidden_dim=enc_hidden, depth=mlp_depth)
        self.decoder = PatchMLPDecoder(embed_dim, self.patch_dim, hidden_dim=dec_hidden, depth=mlp_depth)

        self.vq = EMAVectorQuantizer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            beta=vq_beta,
            ema_decay=ema_decay,
            ema_eps=ema_eps,
            kmeans_init=kmeans_init,
            use_cosine=use_cosine,
        )

    def _pad_to_multiple(self, x: torch.Tensor):
        B, C, H, W = x.shape
        P = self.patch_size
        pad_h = (P - (H % P)) % P
        pad_w = (P - (W % P)) % P
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (pad_h, pad_w)

    def _unpad(self, x: torch.Tensor, pad_hw):
        pad_h, pad_w = pad_hw
        if pad_h == 0 and pad_w == 0:
            return x
        return x[..., : x.shape[-2] - pad_h, : x.shape[-1] - pad_w]

    def patchify(self, x: torch.Tensor):
        # x: (B,C,H,W) -> patches_flat: (B, patch_dim, Np), plus grid info
        x, pad_hw = self._pad_to_multiple(x)
        B, C, H, W = x.shape
        P = self.patch_size
        gh, gw = H // P, W // P  # grid
        patches = self.unfold(x)  # (B, C*P*P, Np)
        return patches, (H, W), (gh, gw), pad_hw

    def unpatchify(self, patches: torch.Tensor, image_hw):
        # patches: (B, patch_dim, Np) -> (B,C,H,W)
        H, W = image_hw
        P = self.patch_size
        if self.fold is None or self.fold.output_size != (H, W):
            self.fold = nn.Fold(output_size=(H, W), kernel_size=P, stride=P).to(patches.device)
        x = self.fold(patches)  # sums overlaps; here no overlap
        return x

    def encode(self, x: torch.Tensor):
        """
        x -> patch latents -> codes
        returns:
          z_q_patches: (B, Np, D)
          indices: (B, Np)
          qloss: scalar
          meta for reconstruction
        """
        patches, (H, W), (gh, gw), pad_hw = self.patchify(x)  # (B, patch_dim, Np)
        B, Dp, Np = patches.shape

        # (B, patch_dim, Np) -> (B*Np, patch_dim)
        patches_flat = patches.transpose(1, 2).contiguous().view(B * Np, Dp)

        z_e = self.encoder(patches_flat)                 # (B*Np, D)
        z_q, indices, qloss = self.vq.quantize(z_e)      # (B*Np, D), (B*Np,), scalar

        z_q = z_q.view(B, Np, -1)
        indices = indices.view(B, Np)

        meta = dict(image_hw=(H, W), grid_hw=(gh, gw), pad_hw=pad_hw, num_patches=Np)
        return z_q, indices, qloss, meta

    def decode(self, z_q: torch.Tensor, meta: dict):
        """
        z_q: (B, Np, D) -> reconstructed image
        """
        B, Np, D = z_q.shape
        H, W = meta["image_hw"]
        pad_hw = meta["pad_hw"]

        patches_flat = z_q.contiguous().view(B * Np, D)        # (B*Np, D)
        patches_rec_flat = self.decoder(patches_flat)          # (B*Np, patch_dim)

        # back to (B, patch_dim, Np)
        patches_rec = patches_rec_flat.view(B, Np, self.patch_dim).transpose(1, 2).contiguous()
        x_hat = self.unpatchify(patches_rec, (H, W))
        x_hat = self._unpad(x_hat, pad_hw)
        return x_hat, patches_rec

    @torch.no_grad()
    def codes_to_image(self, indices: torch.Tensor, meta: dict):
        """
        indices: (B, Np) -> image recon using codebook only
        """
        B, Np = indices.shape
        z_q = self.vq.indices_to_latents(indices.view(-1)).view(B, Np, -1)
        x_hat, _ = self.decode(z_q, meta)
        return x_hat

    def forward(self, x: torch.Tensor, mode: str = "full"):
        """
        mode:
          - "encode": returns (indices, meta)
          - "decode": expects x = (indices, meta) and returns recon image
          - "full": returns (x_hat, qloss, indices, meta, patches_rec)
        """
        if mode == "encode":
            z_q, indices, qloss, meta = self.encode(x)
            return indices, meta

        if mode == "decode":
            indices, meta = x
            return self.codes_to_image(indices, meta)

        if mode == "full":
            z_q, indices, qloss, meta = self.encode(x)
            x_hat, patches_rec = self.decode(z_q, meta)
            return x_hat, qloss, indices, meta, patches_rec

        raise ValueError(f"Unknown mode: {mode}")


# -----------------------------
# Loss / training step helper
# -----------------------------
def calc_codebook_stats(indices: torch.Tensor, vocab_size: int):
    """
    indices: (B, Np)
    returns: usage_frac, perplexity, unique_count
    """
    flat = indices.reshape(-1)
    unique = torch.unique(flat)
    unique_count = unique.numel()
    usage_frac = unique_count / float(vocab_size)

    # perplexity
    hist = torch.bincount(flat, minlength=vocab_size).float()
    p = hist / (hist.sum() + 1e-8)
    entropy = -(p * (p + 1e-8).log()).sum()
    perplexity = entropy.exp()
    return usage_frac, perplexity, unique_count


class PatchVQTrainerMixin:
    """
    Drop-in style training_calc_losses that matches:
      image -> patches -> latent -> code -> latent -> patch recon -> image recon
    """
    def training_calc_losses(self, batch):
        x = batch["gt"].to(self.device)  # (B,C,H,W)

        # full path
        x_hat, qloss, indices, meta, _patches_rec = self.models["net"](x, mode="full")

        # image-space reconstruction loss
        l_img = self.losses["pixel_loss"](x_hat, x)

        # optional LPIPS (typically expects inputs in [-1, 1])
        if ("lpips_loss" in self.losses) and (getattr(self, "weight_lpips", 0.0) > 1e-6):
            l_lpips = self.losses["lpips_loss"](x_hat, x).mean()
        else:
            l_lpips = torch.zeros((), device=x.device)

        # codebook stats (debug/monitoring)
        usage_frac, perplexity, unique_count = calc_codebook_stats(
            indices=indices, vocab_size=self.models["net"].vq.vocab_size
        )

        total = (
            self.lambda_image * l_img
            + self.weight_lpips * l_lpips
            + self.weight_q * qloss
        )

        logs = {
            "total": total.detach(),
            "l_img": l_img.detach(),
            "l_lpips": l_lpips.detach(),
            "qloss": qloss.detach(),
            "cb_usage_frac": torch.tensor(usage_frac, device=x.device),
            "cb_perplexity": perplexity.detach(),
            "cb_unique": torch.tensor(unique_count, device=x.device),
        }
        return total, logs
