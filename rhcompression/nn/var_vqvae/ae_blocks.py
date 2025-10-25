import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
References:
https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py
"""

class Upsample(nn.Module):
    """Nearest-neighbor 2x upsample followed by 3x3 conv (channel-preserving)."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Downsample(nn.Module):
    """2x downsample via strided 3x3 conv, matching common VAE behavior."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        # Right/bottom padding to mimic legacy behavior from original code.
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


class ResBlock(nn.Module):
    """ResNet block with GroupNorm + SiLU and optional channel projection on the skip."""
    def __init__(self, *, in_channels, out_channels=None, dropout):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.act1  = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.act2  = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return self.skip(x) + h


class SelfAttention(nn.Module):
    """Channel-wise self-attention over HxW with 1x1 qkv and output projections."""
    def __init__(self, in_channels):
        super().__init__()
        self.channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.qkv = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / math.sqrt(in_channels)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        qkv = self.qkv(self.norm(x))               # [B, 3C, H, W]
        B, _, H, W = qkv.shape
        C = self.channels
        q, k, v = qkv.view(B, 3, C, H, W).unbind(1)

        # Attention weights
        q = q.view(B, C, H * W).permute(0, 2, 1)   # [B, HW, C]
        k = k.view(B, C, H * W)                    # [B, C, HW]
        attn = torch.bmm(q, k) * self.scale        # [B, HW, HW]
        attn = F.softmax(attn, dim=2)

        # Attend to values
        v = v.view(B, C, H * W)                    # [B, C, HW]
        attn = attn.permute(0, 2, 1)               # [B, HW, HW]
        h = torch.bmm(v, attn).view(B, C, H, W)    # [B, C, H, W]

        return x + self.proj_out(h)

def make_attn(in_channels, using_sa=True):
    """Factory: SelfAttention or Identity."""
    return SelfAttention(in_channels) if using_sa else nn.Identity()

class Encoder(nn.Module):
    """
    Hierarchical encoder: residual stacks with optional attention at the lowest resolution.
    Produces z (or [mu, logvar] if double_z=True) at 1/(2^(len(ch_mult)-1)) spatial scale.
    """
    def __init__(
        self, *, base_channels=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # Input conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # Downsampling stages
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = base_channels * in_ch_mult[i_level]
            block_out = base_channels * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # Output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.act_out  = nn.SiLU(inplace=True)
        out_ch = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        h = self.conv_out(self.act_out(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    """
    Hierarchical decoder: residual stacks with optional attention at the lowest resolution.
    Maps z back to image space (in_channels).
    """
    def __init__(
        self, *, base_channels=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,  # image channels
        z_channels, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # Lowest-resolution channel size
        block_in = base_channels * ch_mult[self.num_resolutions - 1]

        # z to features
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # Upsampling stages
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = base_channels * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)  # keep order consistent with forward loop

        # Output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.act_out  = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.conv_out(self.act_out(self.norm_out(h)))
        return h
