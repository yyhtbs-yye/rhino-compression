import argparse
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def dist_all_reduce_(x: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def extract_patches(
    images: torch.Tensor,
    patch_size: int,
    stride: Optional[int] = None,
) -> torch.Tensor:
    """
    images: [B, C, H, W] float
    returns patches: [B * P, C*patch_size*patch_size]
    """
    if stride is None:
        stride = patch_size
    B, C, H, W = images.shape
    p = patch_size
    if H < p or W < p:
        raise ValueError(f"patch_size={p} must be <= image dims ({H}, {W})")

    # [B, C, H_out, W_out, p, p]
    patches = images.unfold(2, p, stride).unfold(3, p, stride)
    # [B, H_out, W_out, C, p, p]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # [B, P, D]
    patches = patches.view(B, -1, C * p * p)
    # flatten to [B*P, D]
    return patches.view(-1, patches.shape[-1])


def sample_patches_per_image(
    images: torch.Tensor,
    patch_size: int,
    stride: Optional[int],
    max_patches_per_image: Optional[int],
) -> torch.Tensor:
    """
    Returns [N, D] patch vectors, optionally sub-sampling per image to control compute.
    """
    if stride is None:
        stride = patch_size

    B, C, H, W = images.shape
    p = patch_size

    patches = images.unfold(2, p, stride).unfold(3, p, stride)  # [B,C,Ho,Wo,p,p]
    Ho, Wo = patches.shape[2], patches.shape[3]
    P = Ho * Wo

    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, P, C * p * p)  # [B,P,D]

    if max_patches_per_image is None or max_patches_per_image >= P:
        return patches.reshape(B * P, -1)

    # random subsample per image: pick m indices in [0, P)
    m = max_patches_per_image
    idx = torch.randint(0, P, (B, m), device=patches.device)
    sampled = torch.gather(patches, dim=1, index=idx.unsqueeze(-1).expand(B, m, patches.size(-1)))
    return sampled.reshape(B * m, -1)


# ----------------------------
# EMA Vector Quantizer (no gradients needed)
# ----------------------------

class EMAVectorQuantizer(nn.Module):
    """
    EMA-based codebook updates (VQ-VAE style), but usable standalone as online k-means:
      - Assign each vector to nearest code
      - Update codebook with EMA of assigned sums and counts

    Typical usage for standalone training:
      vq = EMAVectorQuantizer(K, D)
      vq.init_from_data(x_batch)
      for x in data:
          stats = vq.update(x)
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
        reinit_dead_codes: bool = True,
        nn_search_chunk_size: int = 131072,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.decay = float(decay)
        self.eps = float(eps)
        self.dead_code_threshold = float(dead_code_threshold)
        self.reinit_dead_codes = bool(reinit_dead_codes)
        self.nn_search_chunk_size = int(nn_search_chunk_size)

        # Buffers hold codebook and EMA accumulators
        embed = torch.empty(self.num_codes, self.code_dim, device=device, dtype=dtype)
        nn.init.uniform_(embed, -1.0 / self.num_codes, 1.0 / self.num_codes)

        self.register_buffer("embedding", embed)                          # [K, D]
        self.register_buffer("cluster_size", torch.zeros(self.num_codes, device=device, dtype=dtype))  # [K]
        self.register_buffer("ema_w", embed.clone())                      # [K, D]
        self.register_buffer("_inited", torch.tensor(0, device=device, dtype=torch.int32))

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor) -> None:
        """
        Initialize embedding from a batch of data by random sampling.
        x: [N, D]
        """
        if self._inited.item() == 1:
            return
        if x.dim() != 2 or x.size(1) != self.code_dim:
            raise ValueError(f"x must be [N, {self.code_dim}]")

        N = x.size(0)
        if N < self.num_codes:
            # repeat if batch smaller than K
            reps = math.ceil(self.num_codes / N)
            x_pool = x.repeat(reps, 1)
            idx = torch.randperm(x_pool.size(0), device=x.device)[: self.num_codes]
            init = x_pool[idx]
        else:
            idx = torch.randperm(N, device=x.device)[: self.num_codes]
            init = x[idx]

        self.embedding.copy_(init)
        self.ema_w.copy_(init)
        self.cluster_size.zero_()
        self._inited.fill_(1)

    @torch.no_grad()
    def _nearest_code_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D] -> indices: [N]
        Uses chunking to avoid huge NxK allocations.
        """
        if x.dim() != 2 or x.size(1) != self.code_dim:
            raise ValueError(f"x must be [N, {self.code_dim}]")

        K = self.num_codes
        E = self.embedding  # [K, D]

        # Precompute ||e||^2
        e2 = (E * E).sum(dim=1)  # [K]

        N = x.size(0)
        out = torch.empty(N, device=x.device, dtype=torch.long)

        chunk = self.nn_search_chunk_size
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            xc = x[start:end]  # [n, D]
            # dist(x,e)^2 = ||x||^2 + ||e||^2 - 2 x.e
            x2 = (xc * xc).sum(dim=1, keepdim=True)  # [n,1]
            # [n,K]
            dots = xc @ E.t()
            dists = x2 + e2.unsqueeze(0) - 2.0 * dots
            out[start:end] = torch.argmin(dists, dim=1)
        return out

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [N, D] -> (x_q: [N, D], indices: [N])
        """
        idx = self._nearest_code_indices(x)
        x_q = self.embedding.index_select(0, idx)
        return x_q, idx

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> dict:
        """
        Perform one EMA update step using batch vectors x: [N, D]
        Returns metrics dict (recon_mse, perplexity, usage, dead_codes).
        """
        if self._inited.item() != 1:
            self.init_from_data(x)

        x_q, idx = self.quantize(x)

        # Stats
        recon_mse = F.mse_loss(x_q, x, reduction="mean").item()

        # One-hot counts: [K]
        K = self.num_codes
        counts = torch.bincount(idx, minlength=K).to(dtype=self.cluster_size.dtype)
        # Sums per code: [K, D]
        sums = torch.zeros_like(self.ema_w)
        sums.index_add_(0, idx, x)

        # DDP: sum counts/sums across workers before EMA
        dist_all_reduce_(counts)
        dist_all_reduce_(sums)

        # EMA updates
        self.cluster_size.mul_(self.decay).add_(counts * (1.0 - self.decay))
        self.ema_w.mul_(self.decay).add_(sums * (1.0 - self.decay))

        # Normalize with Laplace smoothing (keeps codes alive)
        n = self.cluster_size.sum()
        # Smooth cluster sizes, then normalize vectors
        smoothed = (self.cluster_size + self.eps) / (n + K * self.eps) * n
        self.embedding.copy_(self.ema_w / smoothed.unsqueeze(1))

        # Optional: re-init dead codes to random batch samples
        dead = (self.cluster_size < self.dead_code_threshold)
        dead_codes = int(dead.sum().item())
        if self.reinit_dead_codes and dead_codes > 0 and x.size(0) > 0:
            # pick random samples to replace dead codes
            replace = dead.nonzero(as_tuple=False).squeeze(1)
            rnd = x[torch.randint(0, x.size(0), (replace.numel(),), device=x.device)]
            self.embedding.index_copy_(0, replace, rnd)
            self.ema_w.index_copy_(0, replace, rnd)
            # bump cluster_size so they don't immediately look dead
            self.cluster_size.index_fill_(0, replace, self.dead_code_threshold)

        # Perplexity & usage (from this batch assignments, not EMA)
        probs = counts / (counts.sum() + 1e-12)
        perplexity = float(torch.exp(-(probs * (probs + 1e-12).log()).sum()).item())
        usage = float((counts > 0).float().mean().item())

        return {
            "recon_mse": recon_mse,
            "perplexity": perplexity,
            "usage": usage,
            "dead_codes": dead_codes,
        }


# ----------------------------
# Training
# ----------------------------

@dataclass
class TrainConfig:
    data_root: str
    image_size: int = 224
    patch_size: int = 8
    stride: Optional[int] = None
    max_patches_per_image: int = 64

    num_codes: int = 1024
    decay: float = 0.99
    eps: float = 1e-5
    dead_code_threshold: float = 1.0
    reinit_dead_codes: bool = True
    nn_search_chunk_size: int = 131072

    batch_size: int = 64
    num_workers: int = 8
    epochs: int = 1
    seed: int = 0

    save_path: str = "ema_codebook.pt"
    log_every: int = 50


def build_imagenet_train_loader(cfg: TrainConfig) -> DataLoader:
    train_dir = os.path.join(cfg.data_root, "train")
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # [0,1]
        # NOTE: no ImageNet mean/std normalization here on purpose,
        # since we want the codebook to model raw RGB patches.
    ])
    ds = datasets.ImageFolder(train_dir, transform=tfm)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=not is_dist(),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/ImageNet/ILSVRC2012_train")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--max_patches_per_image", type=int, default=64)

    p.add_argument("--num_codes", type=int, default=8192)
    p.add_argument("--decay", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--dead_code_threshold", type=float, default=1.0)
    p.add_argument("--reinit_dead_codes", action="store_true")
    p.add_argument("--no_reinit_dead_codes", action="store_true")
    p.add_argument("--nn_search_chunk_size", type=int, default=131072)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--save_path", type=str, default="ema_codebook.pt")
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()

    reinit = True
    if args.reinit_dead_codes:
        reinit = True
    if args.no_reinit_dead_codes:
        reinit = False
    
    reinit = True

    cfg = TrainConfig(
        data_root=args.data_root,
        image_size=args.image_size,
        patch_size=args.patch_size,
        stride=args.stride,
        max_patches_per_image=args.max_patches_per_image,
        num_codes=args.num_codes,
        decay=args.decay,
        eps=args.eps,
        dead_code_threshold=args.dead_code_threshold,
        reinit_dead_codes=reinit,
        nn_search_chunk_size=args.nn_search_chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        seed=args.seed,
        save_path=args.save_path,
        log_every=args.log_every,
    )

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = build_imagenet_train_loader(cfg)

    code_dim = 3 * cfg.patch_size * cfg.patch_size
    vq = EMAVectorQuantizer(
        num_codes=cfg.num_codes,
        code_dim=code_dim,
        decay=cfg.decay,
        eps=cfg.eps,
        dead_code_threshold=cfg.dead_code_threshold,
        reinit_dead_codes=cfg.reinit_dead_codes,
        nn_search_chunk_size=cfg.nn_search_chunk_size,
        device=device,
        dtype=torch.float32,
    ).to(device)

    step = 0
    for epoch in range(cfg.epochs):
        for images, _ in loader:
            images = images.to(device, non_blocking=True)

            # [N, D] patch vectors
            x = sample_patches_per_image(
                images=images,
                patch_size=cfg.patch_size,
                stride=cfg.stride,
                max_patches_per_image=cfg.max_patches_per_image,
            )

            stats = vq.update(x)

            if step % cfg.log_every == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"mse={stats['recon_mse']:.6f} "
                    f"perplexity={stats['perplexity']:.2f} "
                    f"usage={stats['usage']*100:.1f}% "
                    f"dead={stats['dead_codes']}"
                )
            step += 1

    ckpt = {
        "config": asdict(cfg),
        "num_codes": vq.num_codes,
        "code_dim": vq.code_dim,
        "embedding": vq.embedding.detach().cpu(),
        "cluster_size": vq.cluster_size.detach().cpu(),
        "ema_w": vq.ema_w.detach().cpu(),
    }
    torch.save(ckpt, cfg.save_path)
    print(f"Saved codebook to: {cfg.save_path}")


if __name__ == "__main__":
    main()
