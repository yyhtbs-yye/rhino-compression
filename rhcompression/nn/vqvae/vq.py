# vq.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from rhcompression.nn.utils.nearest_neighbour import get_idx

class VectorQuantizer(nn.Module):
    """
    Vanilla VQ (codebook + optional STE).
    forward(x, *, quantize=False, dequantize=False, ste=True,
            return_idx=True, return_loss=True, return_stats=False) -> dict

    When quantize=True:
      returns keys:
        - 'z_quant' : [B,C,h,w]
        - (optional) 'idx_hw' : [B,h,w] long
        - (optional) 'vq_loss' : scalar
        - (optional) 'stats' : dict with:
              'code_hist'       : [V] float (counts per code, per-batch)
              'num_assignments' : int (B*h*w)
              'perplexity'      : scalar tensor
              'dead_code_rate'  : scalar tensor
              'codes_used'      : int tensor (#nonzero bins)
    """

    def __init__(self, vocab_size, embed_dim, use_znorm, beta=0.25):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.use_znorm = bool(use_znorm)
        self.beta = float(beta)
        self.codebook = nn.Embedding(self.vocab_size, self.embed_dim)

    @torch.no_grad()
    def init_embedding(self, std_or_span: float = 0.0):
        if std_or_span > 0:
            nn.init.trunc_normal_(self.codebook.weight, std=std_or_span)
        elif std_or_span < 0:
            s = abs(std_or_span) / self.vocab_size
            self.codebook.weight.uniform_(-s, s)

    def forward(self, x, mode='encode'):

        if mode == 'encode':
            B, C, H, W = x.shape
            x = rearrange(x, "b c h w -> (b h w) c")
            c = get_idx(x, self.codebook.weight, use_znorm=self.use_znorm)  # [N]
            self.bhw = (B, H, W)
            return c
        elif mode == 'decode':
            B, H, W = self.bhw
            y = rearrange(self.codebook(x), "(b h w) c -> b c h w", b=B, h=H, w=W)
            return y
        else:
            raise "Mode no implemented."