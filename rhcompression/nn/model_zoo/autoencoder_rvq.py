import torch
import torch.nn as nn
import torch.nn.functional as F
from rhcompression.nn.blocks.compvis.encdec import Encoder, Decoder
from rhcompression.nn.utils.nearest_neighbour import get_idx

class AutoencoderRVQ(nn.Module):
    """
    Residual / hierarchical VQ-AE (like VAR's VQVAE):
      - L residual quantizers in sequence: r_0 = z, q_l = CB_l(r_{l-1}), r_l = r_{l-1} - q_l.detach()
      - Sum quantized: q = sum_l q_l, with a single ST estimator: z_q = z + (q - z).detach()
      - EMA-updated codebooks (no codebook loss; commitment only), optional k-means init per level

    vqconfig knobs (all optional; defaults shown):
      num_levels        (4)     # number of residual codebooks
      beta              (0.25)  # float or list[float] per level (commitment weights)
      ema_decay         (0.99)
      ema_eps           (1e-5)
      kmeans_init       (True)
      kmeans_iters      (10)
      kmeans_subsample  (131072)
      kmeans_pp         (False)
      use_znorm         (True)  # pass-through to get_idx

    Returns:
      encode(x): (z_q, indices, qloss)
        - z_q:        (B, C, H, W) quantized latent (post-ST)
        - indices:    (L, B, H, W) integer codes per level
        - qloss:      scalar commitment loss (sum over levels)
    """
    def __init__(self, ddconfig, vqconfig, embed_dim, vocab_size):
        super().__init__()
        zc = ddconfig["z_channels"]

        # encoder / decoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"] == False

        # z_channels -> embed_dim (pre-quant) and back
        self.quant_conv = nn.Conv2d(zc, embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, zc, 1)

        # hyperparams
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # hierarchy controls
        self.num_levels = int(vqconfig.get("num_levels", 4))
        self.use_znorm = bool(vqconfig.get("use_znorm", True))

        # per-level beta (commitment)
        beta = vqconfig.get("beta", 0.25)
        if isinstance(beta, (list, tuple)):
            assert len(beta) == self.num_levels, "beta list must match num_levels"
            self.beta = [float(b) for b in beta]
        else:
            self.beta = [float(beta)] * self.num_levels

        # EMA + kmeans shared knobs
        self.ema_decay = float(vqconfig.get("ema_decay", 0.99))
        self.ema_eps = float(vqconfig.get("ema_eps", 1e-5))
        self.kmeans_init = bool(vqconfig.get("kmeans_init", True))
        self.kmeans_iters = int(vqconfig.get("kmeans_iters", 10))
        self.kmeans_subsample = int(vqconfig.get("kmeans_subsample", 131072))
        self.kmeans_pp = bool(vqconfig.get("kmeans_pp", False))

        # build residual codebooks
        self.codebooks = nn.ModuleList([
            _EMACodebook(
                embed_dim=embed_dim,
                vocab_size=vocab_size,
                beta=self.beta[i],
                ema_decay=self.ema_decay,
                ema_eps=self.ema_eps,
                kmeans_init=self.kmeans_init,
                kmeans_iters=self.kmeans_iters,
                kmeans_subsample=self.kmeans_subsample,
                kmeans_pp=self.kmeans_pp,
                use_znorm=self.use_znorm,
            )
            for i in range(self.num_levels)
        ])

    # ---------- core ops ----------
    def _quantize_hierarchical(self, z):
        """
        Residual quantization over L codebooks.
        Input:
          z: (B, C==embed_dim, H, W)
        Returns:
          z_q_st:   (B,C,H,W) summed quantized with ST to z
          all_idx:  (L,B,H,W) indices per level
          qloss:    scalar
        """
        B, C, H, W = z.shape
        residual = z
        q_sum = torch.zeros_like(z)
        all_indices = []
        qloss = z.new_zeros(())

        for cb in self.codebooks:
            z_q_l, idx_l, qloss_l = cb.quantize(residual, training=self.training)
            q_sum = q_sum + z_q_l
            # residual path is stop-grad so only the last ST affects encoder
            residual = residual - z_q_l.detach()
            all_indices.append(idx_l)
            qloss = qloss + qloss_l

        # single straight-through estimator at the end
        z_q_st = z + (q_sum - z).detach()
        all_indices = torch.stack(all_indices, dim=0)  # (L,B,H,W)
        return z_q_st, all_indices, qloss

    # ---------- public API ----------
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)  # (B, embed_dim, H, W)
        z_q, indices, qloss = self._quantize_hierarchical(h)
        return z_q, indices, qloss

    def decode(self, z_q):
        z = self.post_quant_conv(z_q)
        return self.decoder(z)

    @torch.no_grad()
    def indices_to_latents(self, indices, shape):
        """
        indices: (L,B,H,W) or list/tuple of L tensors (B,H,W)
        shape:   (B, embed_dim, H, W)  -> purely for size sanity
        Returns summed latent (B,C,H,W) from all levels.
        """
        B, C, H, W = shape
        if isinstance(indices, (list, tuple)):
            assert len(indices) == self.num_levels
            per_level = indices
        else:
            assert indices.dim() == 4 and indices.size(0) == self.num_levels
            per_level = [indices[l] for l in range(self.num_levels)]

        z_sum = torch.zeros(B, C, H, W, device=self.codebooks[0].embedding.device)
        for l, cb in enumerate(self.codebooks):
            idx = per_level[l].contiguous().view(-1)  # (B*H*W,)
            z_l = cb.embedding[idx]                   # (B*H*W, C)
            z_l = z_l.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_sum.add_(z_l)
        return z_sum

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
            return self.encode(x)


class _EMACodebook(nn.Module):
    """
    One EMA codebook (dictionary) with optional k-means(+/++) lazy init.
    Quantizes a residual tensor (B,C,H,W) and returns:
      z_q:    (B,C,H,W)
      idx:    (B,H,W)
      qloss:  scalar commitment term
    """
    def __init__(self,
                 embed_dim, vocab_size, beta,
                 ema_decay=0.99, ema_eps=1e-5,
                 kmeans_init=True, kmeans_iters=10, kmeans_subsample=131072, kmeans_pp=False,
                 use_znorm=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.beta = float(beta)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.kmeans_init = bool(kmeans_init)
        self.kmeans_iters = int(kmeans_iters)
        self.kmeans_subsample = int(kmeans_subsample)
        self.kmeans_pp = bool(kmeans_pp)
        self.use_znorm = bool(use_znorm)

        # codebook + EMA state
        embed = torch.randn(vocab_size, embed_dim)
        embed = F.normalize(embed, dim=1)
        self.embedding = nn.Parameter(embed, requires_grad=False)  # EMA-only

        self.register_buffer("ema_cluster_size", torch.zeros(vocab_size))
        self.register_buffer("ema_weight", torch.zeros(vocab_size, embed_dim))
        self.register_buffer("codebook_initialized", torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def _ema_update(self, z_flat, indices):
        N = z_flat.shape[0]
        K = self.vocab_size

        enc = torch.zeros(N, K, device=z_flat.device)
        enc.scatter_(1, indices.view(-1, 1), 1)

        cluster_size = enc.sum(0)  # (K,)
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size * (1 - self.ema_decay))

        dw = enc.t() @ z_flat  # (K,C)
        self.ema_weight.mul_(self.ema_decay).add_(dw * (1 - self.ema_decay))

        # normalized embedding update
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.ema_eps) / (n + K * self.ema_eps) * n
        embed = self.ema_weight / cluster_size.unsqueeze(1)
        self.embedding.data.copy_(embed)

    @torch.no_grad()
    def _kmeans_plus_plus(self, x, k):
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
        x = z_flat
        if x.size(0) > self.kmeans_subsample:
            idx = torch.randperm(x.size(0), device=x.device)[: self.kmeans_subsample]
            x = x[idx]

        if self.kmeans_pp:
            centers = self._kmeans_plus_plus(x, self.vocab_size)
        else:
            perm = torch.randperm(x.size(0), device=x.device)
            centers = x[perm[: self.vocab_size]].clone()

        # Lloyd's iterations
        for _ in range(self.kmeans_iters):
            z_norm = (x ** 2).sum(dim=1, keepdim=True)       # (M,1)
            c_norm = (centers ** 2).sum(dim=1).unsqueeze(0)  # (1,K)
            dist = z_norm + c_norm - 2.0 * x @ centers.t()   # (M,K)
            labels = dist.argmin(dim=1)                      # (M,)
            for ki in range(self.vocab_size):
                mask = labels == ki
                if mask.any():
                    centers[ki] = x[mask].mean(dim=0)
                else:
                    j = torch.randint(0, x.size(0), (1,), device=x.device)
                    centers[ki] = x[j]

        # set codebook + EMA consistent with centers
        self.embedding.data.copy_(centers)
        self.ema_weight.data.copy_(centers)
        self.ema_cluster_size.data.fill_(1.0)
        self.codebook_initialized.fill_(1)

    def quantize(self, r, training: bool):
        """
        r: (B,C,H,W) residual to quantize
        returns: z_q_l (B,C,H,W), idx (B,H,W), qloss (scalar)
        """
        B, C, H, W = r.shape
        r_perm = r.permute(0, 2, 3, 1).contiguous()   # (B,H,W,C)
        r_flat = r_perm.view(-1, C)                   # (N,C)

        # lazy k-means init (first batch that reaches this level)
        if self.kmeans_init and self.codebook_initialized.item() == 0:
            with torch.no_grad():
                self._kmeans_init_from_batch(r_flat)

        # nearest neighbors
        idx = get_idx(r_flat, self.embedding.data, use_znorm=self.use_znorm)  # (N,)
        z_q_flat = self.embedding[idx]                                        # (N,C)
        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # EMA update (train only)
        if training:
            with torch.no_grad():
                self._ema_update(r_flat, idx)

        # commitment on residual
        qloss = self.beta * F.mse_loss(z_q.detach(), r)

        return z_q, idx.view(B, H, W), qloss
