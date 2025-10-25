"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""

import torch
import torch.nn as nn

from rhcompression.nn.var_vqvae.ae_blocks import Decoder, Encoder
from rhcompression.nn.var_vqvae.vq import MultiScaleVectorQuantizer

class VARVQVAE(nn.Module):
    def __init__(self,
                 vocab_size=4096,
                 embed_dim=32,                  # (was: z_channels)
                 in_channels=3,
                 base_channels=128,
                 dropout=0.0,
                 beta=0.25,                     # commitment loss weight
                 use_znorm=False,               # cosine if True, L2 if False
                 quant_conv_kernel=3,           # (was: quant_conv_ks)
                 residual_ratio=0.5,            # (was: quant_resi) phi(x)=a*conv(x)+(1-a)*x
                 share_residual=4,              # (was: share_quant_resi) partially shared phi
                 default_qresi_counts=0,        # if 0, set to len(patch_nums)
                 patch_nums=(1, 2, 4, 8, 16),   # number of patches at each scale
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Encoder/decoder config (mirrors LDM vq-f16)
        ddconfig = dict(
            dropout=dropout,
            base_channels=base_channels,
            z_channels=embed_dim,
            in_channels=in_channels,
            ch_mult=(1, 1, 2, 2, 4),
            num_res_blocks=2,
            using_sa=True,
            using_mid_sa=True,
        )
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.downsample = 2 ** (len(ddconfig["ch_mult"]) - 1)

        # Vector quantizer
        self.quantize = MultiScaleVectorQuantizer(
            vocab_size=vocab_size,
            embed_dim=self.embed_dim,
            use_znorm=use_znorm,
            beta=beta,
            default_qresi_counts=default_qresi_counts,
            patch_nums=patch_nums,
            residual_ratio=residual_ratio,
            share_residual=share_residual,
        )

        # Pre/post quantization convs
        self.quant_conv = nn.Conv2d(self.embed_dim, self.embed_dim, quant_conv_kernel, stride=1, padding=quant_conv_kernel // 2)
        self.post_quant_conv = nn.Conv2d(self.embed_dim, self.embed_dim, quant_conv_kernel, stride=1, padding=quant_conv_kernel // 2)

    # ===================== Training / DDP-friendly forward =====================
    def forward(
        self,
        x: torch.Tensor,
        *,
        ret_usages: bool = False,
        return_dict: bool = False,
        ret_progressive: bool = False,
        ret_indices: bool = False,
        patch_nums=None,
        clamp: bool = True,
    ):
        """
        Encode -> quantize (STE) -> decode.

        Backward-compat:
          return_dict=False  -> returns (img, usages, vq_loss)

        DDP/Trainer-friendly rich output:
          return_dict=True   -> returns a dict with:
            - 'recon'   : reconstructed image (B,C,H,W)
            - 'z_pre'   : pre-quant features (B,D,h,w)   [a.k.a. encoder features after quant_conv]
            - 'z_quant' : quantized features (B,D,h,w)
            - 'vq_loss' : scalar/codebook loss
            - 'usages'  : (optional) usage stats when ret_usages=True
            - 'prog_recons': (optional list[T]) progressive reconstructions after each scale (when ret_progressive=True)
            - 'indices' : (optional list[LongTensor]) per-scale code indices (when ret_indices=True)
        """
        # encoder -> pre-quant feats
        z_pre = self.quant_conv(self.encoder(x))

        # quantize
        z_quant, usages, vq_loss = self.quantize(z_pre, ret_usages=ret_usages)

        # decode
        recon = self.decoder(self.post_quant_conv(z_quant))
        if clamp:
            recon = recon.clamp_(-1, 1)

        if not return_dict:
            # original (img, usages, vq_loss) API
            return recon, usages, vq_loss

        out = {
            'recon': recon,
            'z_pre': z_pre,
            'z_quant': z_quant,
            'vq_loss': vq_loss,
        }
        if ret_usages:
            out['usages'] = usages

        # Optional extras (avoids extra calls from training loops that only use forward)
        if ret_progressive:
            f_hats = self.quantize.features_to_indices_or_recon(z_pre, to_recon=True, patch_nums=patch_nums)
            prog_recons = [self.decoder(self.post_quant_conv(fh)).clamp_(-1, 1) for fh in f_hats]
            out['prog_recons'] = prog_recons

        if ret_indices:
            ms_indices = self.quantize.features_to_indices_or_recon(z_pre, to_recon=False, patch_nums=patch_nums)
            out['indices'] = ms_indices

        return out

    # ===================== Utility: decode a quantized feature map =====================
    def recon_to_img(self, f_hat):
        """Decode a quantized feature map into image space, clamped to [-1,1]."""
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)

    # ===================== Image -> code indices (per scale) =====================
    def img_to_indices(self, img, patch_nums=None):
        """Encode an image and return a list of index tensors (one per scale)."""
        f = self.quant_conv(self.encoder(img))
        return self.quantize.features_to_indices_or_recon(f, to_recon=False, patch_nums=patch_nums)

    # ===================== Code indices -> image =====================
    def indices_to_img(self, ms_indices, same_shape, last_one=False):
        """
        Decode multi-scale indices to image(s).
        If same_shape=True, upsample all scales to max H/W before summing.
        If last_one=True, return only the final image; else return a list.
        """
        B = ms_indices[0].shape[0]
        ms_feats = []
        for idx in ms_indices:
            L = idx.shape[1]
            pn = round(L ** 0.5)
            ms_feats.append(self.quantize.embedding(idx).transpose(1, 2).view(B, self.embed_dim, pn, pn))
        return self.embeds_to_img(ms_feats, all_to_max_scale=same_shape, last_one=last_one)

    # ===================== Embedding feature maps -> image =====================
    def embeds_to_img(self, ms_feats, all_to_max_scale, last_one=False):
        """
        Decode per-scale embedding feature maps to image(s).
        If last_one=True, return only the final image; else return a list.
        """
        if last_one:
            f_hat = self.quantize.embeds_to_recon(ms_feats, all_to_max_scale=all_to_max_scale, last_one=True)
            return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
        else:
            f_hats = self.quantize.embeds_to_recon(ms_feats, all_to_max_scale=all_to_max_scale, last_one=False)
            return [self.decoder(self.post_quant_conv(f)).clamp_(-1, 1) for f in f_hats]

    # ===================== Robust state loading (handles ema shape mismatch) =====================
    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Align saved ema buffer shape if it differs by #scales before loading."""
        key = "quantize.ema_vocab_hit_SV"
        if key in state_dict and state_dict[key].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict[key] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
