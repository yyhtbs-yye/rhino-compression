import torch
import torch.nn as nn
import torch.nn.functional as F
from rhcompression.nn.blocks.compvis.encdec import Encoder, Decoder

from rhcompression.nn.utils.nearest_neighbour import get_idx

class AutoencoderVQ(nn.Module):
    """
    Single-class VQ-AE with:
      - EMA-updated codebook (no codebook loss; commitment only)
      - Optional k-means codebook initialization on first forward
    Additional knobs can be passed via ddconfig:
      ema_decay (0.99), ema_eps (1e-5),
      beta (0.25),
      kmeans_init (True), kmeans_iters (10), kmeans_subsample (131072), kmeans_pp (False)
    """
    def __init__(self, ddconfig, vqconfig, embed_dim, vocab_size):
        super().__init__()
        zc = ddconfig["z_channels"]

        # encoder / decoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"] == False

        # z_channels -> embed_dim (pre-quant) and embed_dim -> z_channels (post-quant)
        self.quant_conv = nn.Conv2d(zc, embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, zc, 1)

        # hyperparams
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.beta = vqconfig.get("beta", 0.25)
        self.ema_decay = vqconfig.get("ema_decay", 0.99)
        self.ema_eps = vqconfig.get("ema_eps", 1e-5)

        # k-means init controls
        self.kmeans_init = vqconfig.get("kmeans_init", True)
        self.kmeans_iters = vqconfig.get("kmeans_iters", 10)
        self.kmeans_subsample = vqconfig.get("kmeans_subsample", 131072)
        self.kmeans_pp = vqconfig.get("kmeans_pp", False)

        # codebook (dictionary) + EMA state
        embed = torch.randn(vocab_size, embed_dim)
        embed = F.normalize(embed, dim=1)  # stable start
        self.embedding = nn.Parameter(embed, requires_grad=False)  # EMA updates only

        self.register_buffer("ema_cluster_size", torch.zeros(vocab_size))
        self.register_buffer("ema_weight", torch.zeros(vocab_size, embed_dim))
        self.register_buffer("codebook_initialized", torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def _ema_update(self, z_flat, indices):
        # z_flat: (N,C), indices: (N,)
        N = z_flat.shape[0]
        K = self.vocab_size

        # one-hot assignments
        enc = torch.zeros(N, K, device=z_flat.device)
        enc.scatter_(1, indices.view(-1, 1), 1)

        # cluster sizes
        cluster_size = enc.sum(0)  # (K,)
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size * (1 - self.ema_decay))

        # ema weight update
        dw = enc.t() @ z_flat  # (K,C)
        self.ema_weight.mul_(self.ema_decay).add_(dw * (1 - self.ema_decay))

        # normalised embedding
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.ema_eps) / (n + K * self.ema_eps) * n
        embed = self.ema_weight / cluster_size.unsqueeze(1)
        self.embedding.data.copy_(embed)

    @torch.no_grad()
    def _kmeans_plus_plus(self, x, k):
        # x: (N,C), returns (k,C) initial centers
        N = x.size(0)
        centers = torch.empty(k, x.size(1), device=x.device)
        idx = torch.randint(0, N, (1,), device=x.device)
        centers[0] = x[idx]
        closest_dist_sq = ((x - centers[0]) ** 2).sum(dim=1)
        for i in range(1, k):
            probs = closest_dist_sq / closest_dist_sq.sum()
            idx = torch.multinomial(probs, 1)
            centers[i] = x[idx]
            dist_sq = ((x - centers[i]) ** 2).sum(dim=1)
            closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq)
        return centers

    @torch.no_grad()
    def _kmeans_init_from_batch(self, z_flat):
        # subsample for speed/memory
        if z_flat.size(0) > self.kmeans_subsample:
            idx = torch.randperm(z_flat.size(0), device=z_flat.device)[: self.kmeans_subsample]
            x = z_flat[idx]
        else:
            x = z_flat

        # initial centers
        if self.kmeans_pp:
            centers = self._kmeans_plus_plus(x, self.vocab_size)
        else:
            perm = torch.randperm(x.size(0), device=x.device)
            centers = x[perm[: self.vocab_size]].clone()

        # Lloyd's iterations
        for _ in range(self.kmeans_iters):
            # assign
            z_norm = (x ** 2).sum(dim=1, keepdim=True)       # (M,1)
            c_norm = (centers ** 2).sum(dim=1).unsqueeze(0)  # (1,K)
            dist = z_norm + c_norm - 2.0 * x @ centers.t()   # (M,K)
            labels = dist.argmin(dim=1)                      # (M,)

            # recompute centers
            for ki in range(self.vocab_size):
                mask = labels == ki
                if mask.any():
                    centers[ki] = x[mask].mean(dim=0)
                else:
                    # re-seed empty cluster
                    j = torch.randint(0, x.size(0), (1,), device=x.device)
                    centers[ki] = x[j]

        # set codebook and EMA state consistent with init
        self.embedding.data.copy_(centers)
        self.ema_weight.data.copy_(centers)
        self.ema_cluster_size.data.fill_(1.0)
        self.codebook_initialized.fill_(1)

    # ---------- core ops ----------
    def _quantize(self, z):
        # z: (B,C,H,W) with C=embed_dim
        B, C, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
        z_flat = z_perm.view(-1, C)                  # (N,C)

        # lazy k-means init
        if self.kmeans_init and self.codebook_initialized.item() == 0:
            self._kmeans_init_from_batch(z_flat)

        # nearest neighbors
        indices = get_idx(z_flat, self.embedding.data, use_znorm=True)           # (N,)
        z_q_flat = self.embedding[indices]                     # (N,C)
        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2)

        # EMA dictionary update (training only)
        if self.training:
            self._ema_update(z_flat, indices)

        # commitment loss and straight-through
        qloss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()

        # indices to (B,H,W)
        indices = indices.view(B, H, W)
        return z_q, indices, qloss

    # ---------- public API ----------
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        z_q, indices, qloss = self._quantize(h)
        return z_q, indices, qloss

    def decode(self, z_q):
        z = self.post_quant_conv(z_q)
        return self.decoder(z)

    @torch.no_grad()
    def indices_to_latents(self, indices, shape):
        # shape: (B, embed_dim, H, W)
        B, C, H, W = shape
        z_q = self.embedding[indices.view(-1)]
        z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return z_q

    def forward(self, x, mode='encode'):
        if mode == 'encode':
            return self.encode(x)
        elif mode == 'decode':
            return self.decode(x)
        elif mode == 'full':
            z_q, indices, qloss = self.encode(x)
            recon = self.decode(z_q)
            return recon, qloss, indices
        else:
            # default to encode for brevity
            return self.encode(x)
