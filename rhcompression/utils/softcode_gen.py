import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# ----------------------------
# Patch extraction (unfold)
# ----------------------------
def extract_patches(
    images: torch.Tensor,  # [B,C,H,W]
    patch_size: int,
    stride: int,
    max_patches_per_image: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns patches: [N,C,patch_size,patch_size]
    """
    B, C, H, W = images.shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image too small: {H}x{W} < {patch_size}x{patch_size}")

    unfolded = F.unfold(images, kernel_size=patch_size, stride=stride)  # [B, C*ps*ps, L]
    B, CP, L = unfolded.shape
    patches = unfolded.transpose(1, 2).contiguous().view(B, L, C, patch_size, patch_size)

    if max_patches_per_image is not None and max_patches_per_image < L:
        # sample the same set across the batch (fast). If you want per-image, do per B.
        idx = torch.randperm(L, device=images.device)[:max_patches_per_image]
        patches = patches[:, idx]

    return patches.reshape(-1, C, patch_size, patch_size)


# ----------------------------
# Complexity gating
# ----------------------------
def patch_complexity_grad_energy(patches: torch.Tensor) -> torch.Tensor:
    """
    patches: [N,C,P,P]
    returns: [N] complexity score (bigger => more edges/texture)
    """
    dx = patches[..., :, 1:] - patches[..., :, :-1]
    dy = patches[..., 1:, :] - patches[..., :-1, :]
    return dx.abs().mean(dim=(1, 2, 3)) + dy.abs().mean(dim=(1, 2, 3))


# ----------------------------
# Feature builder (no encoder)
# ----------------------------
def patch_features(
    patches: torch.Tensor,          # [N,C,P,P]
    remove_mean: bool = True,
    l2_normalize: bool = False,
) -> torch.Tensor:
    """
    For tiny patches, raw pixels (flatten) work well.
    Common trick: remove per-patch mean so matching focuses on texture/edges.
    """
    x = patches.flatten(1)  # [N,D]
    if remove_mean:
        x = x - x.mean(dim=1, keepdim=True)
    if l2_normalize:
        x = F.normalize(x, dim=1, eps=1e-8)
    return x


# ----------------------------
# Chunked nearest-centroid search
# ----------------------------
@torch.no_grad()
def nearest_centroid(
    x: torch.Tensor,            # [N,D]
    centroids: torch.Tensor,    # [K,D]
    chunk_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      idx: [N]
      d2:  [N] squared L2 to chosen centroid
    """
    N, D = x.shape
    K, Dc = centroids.shape
    assert D == Dc

    c = centroids
    c2 = (c * c).sum(dim=1).view(1, K)  # [1,K]

    idx_out = torch.empty(N, device=x.device, dtype=torch.long)
    d2_out = torch.empty(N, device=x.device, dtype=x.dtype)

    for s in range(0, N, chunk_size):
        t = min(s + chunk_size, N)
        xs = x[s:t]  # [m,D]
        x2 = (xs * xs).sum(dim=1, keepdim=True)          # [m,1]
        d2 = x2 + c2 - 2.0 * (xs @ c.t())                # [m,K]
        d2min, idx = torch.min(d2, dim=1)                # [m]
        idx_out[s:t] = idx
        d2_out[s:t] = d2min

    return idx_out, d2_out


# ----------------------------
# EMA Online K-means (no encoder/decoder)
# ----------------------------
@dataclass
class EMAKMeansCfg:
    k: int
    patch_size: int
    in_ch: int

    decay: float = 0.99
    eps: float = 1e-5

    remove_mean: bool = True
    l2_normalize: bool = False

    complex_thresh: Optional[float] = None  # None disables gating
    nn_chunk_size: int = 8192

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_root: str = "data/ImageNet/ILSVRC2012_train/train"
    image_size: int = 256
    batch_size: int = 64
    num_workers: int = 8

class OnlineEMAKMeans:
    """
    Online EMA updates of centroids:
      cluster_size <- decay*cluster_size + (1-decay)*counts
      embed_sum    <- decay*embed_sum    + (1-decay)*sums
      centroids    <- embed_sum / normalized_cluster_size

    Also tracks true hit counts (not EMA) for usage statistics.
    """
    def __init__(self, cfg: EMAKMeansCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.D = cfg.in_ch * cfg.patch_size * cfg.patch_size

        self.centroids = torch.empty(cfg.k, self.D, device=self.device, dtype=torch.float32)
        self.ema_cluster_size = torch.zeros(cfg.k, device=self.device, dtype=torch.float32)
        self.ema_embed_sum = torch.zeros(cfg.k, self.D, device=self.device, dtype=torch.float32)

        self.hit_count = torch.zeros(cfg.k, device=self.device, dtype=torch.long)  # true counts
        self.codebook_initialized = False

        self.total_seen = 0
        self.total_gated_in = 0

    @torch.no_grad()
    def _init_from_batch(self, x: torch.Tensor):
        """
        Initialize centroids from a batch of features by random sampling.
        """
        N = x.size(0)
        if N < self.cfg.k:
            # repeat samples if not enough
            reps = math.ceil(self.cfg.k / max(1, N))
            x_rep = x.repeat(reps, 1)[: self.cfg.k]
            init = x_rep
        else:
            perm = torch.randperm(N, device=x.device)[: self.cfg.k]
            init = x[perm]

        self.centroids.copy_(init)
        self.ema_embed_sum.copy_(init)
        self.ema_cluster_size.fill_(1.0)
        self.codebook_initialized = True

    @torch.no_grad()
    def update(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: [N,C,P,P] recommended in [-1,1]
        Returns:
          indices: [N] with -1 for gated-out patches (if gating enabled)
        """
        patches = patches.to(self.device, non_blocking=True).float()
        N = patches.size(0)
        self.total_seen += N

        # gating
        if self.cfg.complex_thresh is not None:
            comp = patch_complexity_grad_energy(patches)
            keep = comp >= float(self.cfg.complex_thresh)
        else:
            keep = torch.ones((N,), device=self.device, dtype=torch.bool)

        out_idx = torch.full((N,), -1, device=self.device, dtype=torch.long)
        if keep.sum().item() == 0:
            return out_idx

        patches_k = patches[keep]
        x = patch_features(
            patches_k,
            remove_mean=self.cfg.remove_mean,
            l2_normalize=self.cfg.l2_normalize,
        )

        self.total_gated_in += x.size(0)

        # init centroids on first use
        if not self.codebook_initialized:
            self._init_from_batch(x)

        # assign
        idx, _d2 = nearest_centroid(x, self.centroids, chunk_size=self.cfg.nn_chunk_size)

        # true hit count
        self.hit_count += torch.bincount(idx, minlength=self.cfg.k).to(self.hit_count.dtype)

        # EMA update stats
        counts = torch.bincount(idx, minlength=self.cfg.k).float()  # [K]
        sums = torch.zeros(self.cfg.k, self.D, device=self.device, dtype=torch.float32)
        sums.index_add_(0, idx, x)  # accumulate x per centroid

        # EMA updates
        decay = self.cfg.decay
        self.ema_cluster_size.mul_(decay).add_(counts * (1.0 - decay))
        self.ema_embed_sum.mul_(decay).add_(sums * (1.0 - decay))

        # normalized centroids (Laplace smoothing style)
        n = self.ema_cluster_size.sum()
        K = self.cfg.k
        denom = (self.ema_cluster_size + self.cfg.eps) / (n + K * self.cfg.eps) * n
        self.centroids.copy_(self.ema_embed_sum / denom.unsqueeze(1))

        # write back indices
        out_idx[keep] = idx
        return out_idx

    @torch.no_grad()
    def assign(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign without updating EMA.
        Returns:
          idx: [N] (-1 for gated out if gating on)
          d2:  [N] squared distance (inf for gated out)
        """
        patches = patches.to(self.device, non_blocking=True).float()
        N = patches.size(0)

        if self.cfg.complex_thresh is not None:
            comp = patch_complexity_grad_energy(patches)
            keep = comp >= float(self.cfg.complex_thresh)
        else:
            keep = torch.ones((N,), device=self.device, dtype=torch.bool)

        idx_out = torch.full((N,), -1, device=self.device, dtype=torch.long)
        d2_out = torch.full((N,), float("inf"), device=self.device, dtype=torch.float32)

        if keep.sum().item() == 0:
            return idx_out, d2_out

        patches_k = patches[keep]
        x = patch_features(
            patches_k,
            remove_mean=self.cfg.remove_mean,
            l2_normalize=self.cfg.l2_normalize,
        )

        idx, d2 = nearest_centroid(x, self.centroids, chunk_size=self.cfg.nn_chunk_size)
        idx_out[keep] = idx
        d2_out[keep] = d2
        return idx_out, d2_out


# ----------------------------
# Exemplar (medoid) finding pass
# ----------------------------
@torch.no_grad()
def find_exemplars(
    kmeans: OnlineEMAKMeans,
    loader,                      # yields (imgs, labels) in [0,1]
    steps: int,
    patch_size: int,
    stride: int,
    max_patches_per_image: Optional[int] = None,
    use_y: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Second pass: find an actual patch exemplar for each centroid:
      exemplar[j] = argmin_patch ||feat(patch) - centroid[j]||^2

    Returns dict with:
      exemplar_patches: [K,C,P,P] (float32 in [-1,1])
      exemplar_d2:      [K] best squared distances
      exemplar_found:   [K] bool
      exemplar_hits:    [K] true hit count from training
    """
    device = kmeans.device
    K = kmeans.cfg.k
    C = kmeans.cfg.in_ch
    P = patch_size

    best_d2 = torch.full((K,), float("inf"), device=device, dtype=torch.float32)
    best_patch = torch.zeros((K, C, P, P), device=device, dtype=torch.float32)
    found = torch.zeros((K,), device=device, dtype=torch.bool)

    def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, _ = next(it)

        imgs = imgs.to(device, non_blocking=True)
        if use_y:
            imgs = rgb_to_y(imgs)

        imgs = imgs * 2.0 - 1.0  # [-1,1]

        patches = extract_patches(
            imgs, patch_size=patch_size, stride=stride, max_patches_per_image=max_patches_per_image
        )  # [N,C,P,P]

        idx, d2 = kmeans.assign(patches)  # idx=-1 gated out
        keep = idx >= 0
        if keep.any():
            patches_k = patches[keep]
            idx_k = idx[keep]
            d2_k = d2[keep]

            # Update best exemplar per cluster:
            # For each cluster, keep minimal d2 sample.
            # We do it by sorting by d2 and then first-hit per cluster.
            order = torch.argsort(d2_k)  # ascending
            idx_s = idx_k[order]
            d2_s = d2_k[order]
            p_s = patches_k[order]

            # First occurrence per cluster in sorted list is best for that cluster in this batch
            # We'll scan and update (K can be large, but this is per batch, so OK).
            seen = set()
            for i in range(idx_s.numel()):
                j = int(idx_s[i].item())
                if j in seen:
                    continue
                seen.add(j)
                if d2_s[i] < best_d2[j]:
                    best_d2[j] = d2_s[i]
                    best_patch[j].copy_(p_s[i])
                    found[j] = True

        if step % 50 == 0:
            print(f"[exemplar {step:5d}/{steps}] found {int(found.sum().item())}/{K}")

    return {
        "exemplar_patches": best_patch.detach().cpu(),   # [-1,1]
        "exemplar_d2": best_d2.detach().cpu(),
        "exemplar_found": found.detach().cpu(),
        "hit_count": kmeans.hit_count.detach().cpu(),
        "centroids": kmeans.centroids.detach().cpu(),
        "ema_cluster_size": kmeans.ema_cluster_size.detach().cpu(),
    }


# ----------------------------
# Save / Load
# ----------------------------
def save_codebook(path: str, cfg: EMAKMeansCfg, kmeans: OnlineEMAKMeans, extra: Optional[dict] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    obj = {
        "cfg": cfg.__dict__,
        "state": {
            "centroids": kmeans.centroids.detach().cpu(),
            "ema_cluster_size": kmeans.ema_cluster_size.detach().cpu(),
            "ema_embed_sum": kmeans.ema_embed_sum.detach().cpu(),
            "hit_count": kmeans.hit_count.detach().cpu(),
            "initialized": kmeans.codebook_initialized,
            "stats": {
                "total_seen": kmeans.total_seen,
                "total_gated_in": kmeans.total_gated_in,
            }
        }
    }
    if extra is not None:
        obj["extra"] = extra
    torch.save(obj, path)
    print(f"Saved: {path}")


def load_codebook(path: str, device: Optional[str] = None) -> OnlineEMAKMeans:
    ckpt = torch.load(path, map_location="cpu")
    cfg = EMAKMeansCfg(**ckpt["cfg"])
    if device is not None:
        cfg.device = device
    km = OnlineEMAKMeans(cfg)

    st = ckpt["state"]
    km.centroids.copy_(st["centroids"].to(km.device))
    km.ema_cluster_size.copy_(st["ema_cluster_size"].to(km.device))
    km.ema_embed_sum.copy_(st["ema_embed_sum"].to(km.device))
    km.hit_count.copy_(st["hit_count"].to(km.device))
    km.codebook_initialized = bool(st.get("initialized", True))

    stats = st.get("stats", {})
    km.total_seen = int(stats.get("total_seen", 0))
    km.total_gated_in = int(stats.get("total_gated_in", 0))

    return km

def build_dataloader(data_root: str, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomRotation(90),
        transforms.ToTensor(),  # [0,1]
    ])
    ds = ImageFolder(root=data_root, transform=tfm)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

cfg = EMAKMeansCfg(
    k=8192,
    patch_size=4,
    in_ch=1,
    decay=0.99,
    eps=1e-5,
    remove_mean=True,
    l2_normalize=False,
    complex_thresh=0.20,   # tune (higher => fewer smooth patches)
    nn_chunk_size=8192,
    device="cuda",
    data_root = "data/ImageNet/ILSVRC2012_train/train",
    image_size = 256,
    batch_size = 64,
    num_workers = 8,
)

km = OnlineEMAKMeans(cfg)

steps = 2000
patch_size = cfg.patch_size
stride = patch_size
max_patches_per_image = 256
use_y = True

def rgb_to_y(img):
    r, g, b = img[:,0:1], img[:,1:2], img[:,2:3]
    return 0.299*r + 0.587*g + 0.114*b

loader = build_dataloader(cfg.data_root, cfg.image_size, cfg.batch_size, cfg.num_workers)
it = iter(loader)

for step in range(1, steps + 1):
    try:
        imgs, _ = next(it)
    except StopIteration:
        it = iter(loader)
        imgs, _ = next(it)

    imgs = imgs.to(km.device, non_blocking=True)
    if use_y:
        imgs = rgb_to_y(imgs)

    imgs = imgs * 2.0 - 1.0  # [-1,1]

    patches = extract_patches(imgs, patch_size=patch_size, stride=stride, max_patches_per_image=max_patches_per_image)
    idx = km.update(patches)

    if step % 50 == 0:
        hits = km.hit_count
        used = (hits > 0).float().mean().item() * 100
        top = torch.topk(hits, k=5).values.tolist()
        print(f"[{step:5d}/{steps}] used_codes={used:.1f}% top_hits={top}")

save_codebook("ema_kmeans_centroids.pt", cfg, km)
