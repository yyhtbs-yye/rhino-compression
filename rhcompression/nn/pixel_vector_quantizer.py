# vq_nobp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelVectorQuantizer(nn.Module):
    """
    No-backprop VQ: nearest-neighbor 'assign' and 'embed_code' only.
    EMA updates are expected to be handled by the trainer/boat.
    """
    def __init__(self, num_codes: int, dim: int, eps: float = 1e-5, pretrained: str=None):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.eps = eps

        # codebook and EMA state live here but are updated externally
        self.register_buffer("codebook", torch.randn(num_codes, dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_codebook", torch.zeros(num_codes, dim))

        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, path: str):
        state = torch.load(path, map_location=torch.device('cpu'))
        if 'codebook' in state:
            self.codebook.copy_(state['codebook'])
        if 'ema_cluster_size' in state:
            self.ema_cluster_size.copy_(state['ema_cluster_size'])
        if 'ema_codebook' in state:
            self.ema_codebook.copy_(state['ema_codebook'])
        print(f"Loaded pretrained VQ from {path}")

    @torch.no_grad()
    def kmeans_init(self, z_samples: torch.Tensor, iters: int = 20):
        """
        z_samples: [N, dim] tensor (pre-scale if you quantize pre-scale).
        """
        assert z_samples.dim() == 2 and z_samples.size(1) == self.dim
        N, D = z_samples.shape
        K = self.num_codes
        device = z_samples.device

        # simple random init
        idx = torch.randperm(N, device=device)[:K]
        centers = z_samples[idx].clone()

        for _ in range(iters):
            dists = (z_samples.pow(2).sum(1, keepdim=True)
                     - 2 * z_samples @ centers.t()
                     + centers.pow(2).sum(1)[None, :])
            assign = dists.argmin(1)
            # update centers
            for k in range(K):
                sel = z_samples[assign == k]
                if sel.numel() > 0:
                    centers[k] = sel.mean(0)

        self.codebook.copy_(centers)
        self.ema_codebook.zero_()
        self.ema_cluster_size.zero_()

    @torch.no_grad()
    def assign(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, C==dim, H, W] or [N, dim]
        returns indices:
          - if input was BCHW -> [B, H, W]
          - if input was [N, dim] -> [N]
        """
        if z.dim() == 4:
            B, C, H, W = z.shape
            assert C == self.dim
            zf = z.permute(0,2,3,1).reshape(-1, C)
            flat = True
        else:
            zf = z
            flat = False

        cb = self.codebook
        dists = (zf.pow(2).sum(1, keepdim=True)
                 - 2 * zf @ cb.t()
                 + cb.pow(2).sum(1)[None, :])
        idx = dists.argmin(1)

        if z.dim() == 4:
            return idx.view(B, 1, H, W)
        return idx  # [N]

    @torch.no_grad()
    def embed_code(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, H, W] or [N]
        returns z_q:
          - if [B,H,W] -> [B, C, H, W]
          - if [N] -> [N, C]
        """
        if indices.dim() == 4:
            B, _, H, W = indices.shape
            zq = F.embedding(indices.view(-1), self.codebook).view(B, H, W, self.dim).permute(0,3,1,2)
        elif indices.dim() == 3:
            B, H, W = indices.shape
            zq = F.embedding(indices.reshape(-1), self.codebook).view(B, H, W, self.dim).permute(0,3,1,2)
        else:
            zq = F.embedding(indices, self.codebook)
        return zq

if __name__ == "__main__":
    # simple test
    vq = PixelVectorQuantizer(num_codes=4096, dim=4, pretrained='pretrained/pixel_vq_model_d4k_ffhq256.pt')
    x = torch.randn(2, 4, 16, 16)
    codes = vq.assign(x)
    xq = vq.embed_code(codes)
    print("Input shape:", x.shape)
    print("Codes shape:", codes.shape)
    print("Reconstructed shape:", xq.shape)
    print("Codebook shape:", vq.codebook.shape)