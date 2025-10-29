import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiScaleVectorQuantizer(nn.Module):
    """
    Multi-scale vector quantizer with optional residual refinement filters.
    """

    # VQGAN originally used beta=1.0; Stable Diffusion typically uses 0.25.
    def __init__(
        self,
        vocab_size,
        embed_dim,                  # (was: Cvae) latent/channel dimension
        use_znorm,                  # (was: using_znorm) cosine if True, L2 if False
        beta=0.25,
        default_qresi_counts=0,
        patch_nums=None,           # (was: v_patch_nums) list of side lengths per scale (small -> large)
        residual_ratio=0.5,         # (was: quant_resi)
        share_residual=4,           # (was: share_quant_resi)
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.use_znorm = use_znorm
        self.patch_nums = patch_nums

        self.residual_ratio = residual_ratio
        if share_residual == 0:   # non-shared: separate phi for each scale
            count = default_qresi_counts or len(self.patch_nums)
            self.quant_resi = PhiNonShared([
                (Phi(embed_dim, residual_ratio) if abs(residual_ratio) > 1e-6 else nn.Identity())
                for _ in range(count)
            ])
        elif share_residual == 1: # fully shared: one phi for all scales
            self.quant_resi = PhiShared(
                Phi(embed_dim, residual_ratio) if abs(residual_ratio) > 1e-6 else nn.Identity()
            )
        else:                      # partially shared: several shared phis
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([
                (Phi(embed_dim, residual_ratio) if abs(residual_ratio) > 1e-6 else nn.Identity())
                for _ in range(share_residual)
            ]))

        self.register_buffer(
            'ema_vocab_hit_SV',
            torch.full((len(self.patch_nums), self.vocab_size), fill_value=0.0)
        )
        self.record_hit = 0

        self.beta = beta
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # Progressive training placeholder (not supported; always -1).
        self.prog_si = -1

    def init_embedding(self, std_or_span):
        """Init codebook: >0 trunc_normal(std), <0 uniform(-|x|/V, |x|/V), 0 = no-op."""
        if std_or_span > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=std_or_span)
        elif std_or_span < 0:
            s = abs(std_or_span) / self.vocab_size
            self.embedding.weight.data.uniform_(-s, s)

    def extra_repr(self):
        """Readable config summary."""
        return f'{self.patch_nums}, znorm={self.use_znorm}, beta={self.beta}  |  S={len(self.patch_nums)}, quant_resi={self.residual_ratio}'

    # ===================== Training forward (VAE) =====================
    def forward(self, x, ret_usages=False):
        """
        Quantize-and-reconstruct features with STE; optionally report code usage.

        Returns:
            y          : quantized+residual reconstruction (same shape as x)
            usages     : list of % usage per scale (or None)
            mean_vq_loss: scalar loss (codebook + commit)
        """
        if x.dtype != torch.float32:
            x = x.float()

        B, C, H, W = x.shape
        x_ng = x.detach() # x_ng: no gradient

        resid = x_ng.clone()
        recon = torch.zeros_like(resid)

        mean_vq_loss = torch.tensor(0.0, device=x.device)
        vocab_hits = torch.zeros(self.vocab_size, dtype=torch.float, device=x.device)

        S = len(self.patch_nums)
        emb = self.embedding.weight  # [V, C]

        for si, pn in enumerate(self.patch_nums):  # small -> large
            if si != S - 1:
                z = F.interpolate(resid, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C)
            else:
                z = resid.permute(0, 2, 3, 1).reshape(-1, C)

            if self.use_znorm:
                z_norm = F.normalize(z, dim=-1)
                e_normT = F.normalize(emb, dim=1).t()
                idx = torch.argmax(z_norm @ e_normT, dim=1)
            else:
                x2 = z.pow(2).sum(dim=1, keepdim=True)   # [N,1]
                e2 = emb.pow(2).sum(dim=1)               # [V]
                xe = z @ emb.t()                         # [N,V]
                d = x2 + e2 - 2.0 * xe
                idx = torch.argmin(d, dim=1)

            hits = idx.bincount(minlength=self.vocab_size).float()
            vocab_hits.add_(hits)

            idx_Bhw = idx.view(B, pn, pn)
            h = self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()  # [B,C,pn,pn]
            if si != S - 1:
                h = F.interpolate(h, size=(H, W), mode='bicubic').contiguous()

            # h = self.quant_resi[si](h)
            h = self.quant_resi[si / (S - 1)](h)

            commit = F.mse_loss(h, resid.detach()) * self.beta   # encoder (commitment)
            codebk = F.mse_loss(h.detach(), resid)               # codebook
            mean_vq_loss = mean_vq_loss + (commit + codebk)

            recon = recon + h
            resid = resid - h

        mean_vq_loss = mean_vq_loss * (1.0 / S)

        # STE: forward uses recon; backward passes gradient as identity wrt x.
        y = x + (recon - x).detach()

        if ret_usages:
            usages = []
            for si, pn in enumerate(self.patch_nums):
                margin = (pn * pn) / 100.0
                usage_pct = (self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100.0
                usages.append(usage_pct)
        else:
            usages = None

        return y, usages, mean_vq_loss

    # ===================== Utility: embeddings -> running recon maps =====================
    def embeds_to_recon(self, ms_feats, all_to_max_scale=True, last_one=False):
        """
        Sum per-scale decoded embeddings into reconstructions.
        If all_to_max_scale=True, upsample every scale to max H/W before summing.
        """
        outs = []
        B = ms_feats[0].shape[0]
        H = W = self.patch_nums[-1]
        S = len(self.patch_nums)

        if all_to_max_scale:
            recon = ms_feats[0].new_zeros(B, self.embed_dim, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.patch_nums):  # small -> large
                h = ms_feats[si]
                if si < len(self.patch_nums) - 1:
                    h = F.interpolate(h, size=(H, W), mode='bicubic')
                h = self.quant_resi[si / (S - 1)](h)
                recon.add_(h)
                if last_one:
                    outs = recon
                else:
                    outs.append(recon.clone())
        else:
            # Experimental: progressively grow recon size per scale.
            recon = ms_feats[0].new_zeros(B, self.embed_dim, self.patch_nums[0], self.patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.patch_nums):  # small -> large
                recon = F.interpolate(recon, size=(pn, pn), mode='bicubic')
                h = self.quant_resi[si / (S - 1)](ms_feats[si])
                recon.add_(h)
                if last_one:
                    outs = recon
                else:
                    outs.append(recon)

        return outs

    # ===================== Features -> indices or running recon =====================
    def features_to_indices_or_recon(self, x, to_recon, patch_nums=None):
        """
        Greedily quantize per scale; return list of recon maps (if to_recon)
        or list of flat index tensors per scale (if not).
        """
        B, C, H, W = x.shape
        x_ng = x.detach()
        resid = x_ng.clone()
        recon = torch.zeros_like(resid)

        out = []

        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (patch_nums or self.patch_nums)]
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'patch_hws[-1]={patch_hws[-1]} != (H={H}, W={W})'

        S = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):  # small -> large
            if 0 <= self.prog_si < si:
                break  # progressive (unused)

            z = F.interpolate(resid, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != S - 1) \
                else resid.permute(0, 2, 3, 1).reshape(-1, C)

            if self.use_znorm:
                z = F.normalize(z, dim=-1)
                idx = torch.argmax(z @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d = torch.sum(z.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1)
                d.addmm_(z, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx = torch.argmin(d, dim=1)

            idx_Bhw = idx.view(B, ph, pw)
            h = self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous() if (si == S - 1) \
                else F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous()

            h = self.quant_resi[si / (S - 1)](h)
            recon.add_(h)
            resid.sub_(h)

            out.append(recon.clone() if to_recon else idx.reshape(B, ph * pw))

        return out

    # ===================== VAR teacher-forcing input (training) =====================
    def indices_to_var_input(self, gt_ms_indices):
        """
        Build teacher-forcing inputs across scales by decoding indices and
        providing next-scale pooled features concatenated along sequence dim.
        """
        next_scales = []
        B = gt_ms_indices[0].shape[0]
        C = self.embed_dim
        H = W = self.patch_nums[-1]
        S = len(self.patch_nums)

        recon = gt_ms_indices[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next = self.patch_nums[0]
        for si in range(S - 1):
            if self.prog_si == 0 or (0 <= self.prog_si - 1 < si):
                break  # progressive (unused)
            h = F.interpolate(
                self.embedding(gt_ms_indices[si]).transpose_(1, 2).view(B, C, pn_next, pn_next),
                size=(H, W), mode='bicubic'
            )
            recon.add_(self.quant_resi[si / (S - 1)](h))
            pn_next = self.patch_nums[si + 1]
            next_scales.append(F.interpolate(recon, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))

        return torch.cat(next_scales, dim=1) if len(next_scales) else None  # B x L x C (float32)

    # ===================== VAR inference: make next-step input =====================
    def next_ar_input(self, si, S, recon, h_BChw):
        """
        Update running recon and return (updated_recon, next_scale_feature_map).
        """
        HW = self.patch_nums[-1]
        if si != S - 1:
            h = self.quant_resi[si / (S - 1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))
            recon.add_(h)
            return recon, F.interpolate(recon, size=(self.patch_nums[si + 1], self.patch_nums[si + 1]), mode='area')
        else:
            h = self.quant_resi[si / (S - 1)](h_BChw)
            recon.add_(h)
            return recon, recon


class Phi(nn.Conv2d):
    """Residual refinement conv: y = (1-a)*x + a*conv3x3(x)."""
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, x):
        return x.mul(1 - self.resi_ratio) + super().forward(x).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    """Single shared residual module for all scales."""
    def __init__(self, qresi):
        super().__init__()
        self.qresi = qresi

    def __getitem__(self, _):
        return self.qresi


class PhiPartiallyShared(nn.Module):
    """A few shared residual modules, picked by nearest normalized scale tick."""
    def __init__(self, qresi_ls):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_0_to_1):
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_0_to_1)).item()]

    def extra_repr(self):
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    """One residual module per scale, picked by nearest normalized scale tick."""
    def __init__(self, qresi):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_0_to_1):
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_0_to_1)).item())

    def extra_repr(self):
        return f'ticks={self.ticks}'
