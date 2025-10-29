import torch
import torch.nn as nn
from rhcompression.nn.blocks.compvis.encdec import Encoder, Decoder

class Autoencoder(nn.Module):
    def __init__(self, ddconfig, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"] == False
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        self.reset_parameters()

    def reset_parameters(self, zero_final_decoder: bool = True):
        """
        Initialize weights & biases in a CompVis-friendly way.

        - Convs/ConvTranspose: kaiming_normal_ (fan_out), bias=0
        - Linear: xavier_uniform_, bias=0
        - Norms (BatchNorm/GroupNorm/LayerNorm): weight=1, bias=0
        - 1x1 quant/post-quant convs: identity if shapes match else xavier_uniform_
        - Optionally zero the last Conv2d in the decoder.
        """
        # Generic init helpers
        def init_conv(m: nn.Module):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Apply to everything first
        self.apply(init_conv)

        # Identity-initialize the 1x1 adapters when possible
        def try_identity_1x1(conv: nn.Conv2d):
            if isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1):
                Cout, Cin = conv.out_channels, conv.in_channels
                if Cout == Cin:
                    with torch.no_grad():
                        conv.weight.zero_()
                        # Weight tensor shape: (Cout, Cin, 1, 1)
                        eye = torch.eye(Cout, device=conv.weight.device, dtype=conv.weight.dtype).view(Cout, Cout, 1, 1)
                        conv.weight.copy_(eye)
                        if conv.bias is not None:
                            conv.bias.zero_()
                    return True
            return False

        # quant_conv: z_channels -> embed_dim
        if not try_identity_1x1(self.quant_conv):
            nn.init.xavier_uniform_(self.quant_conv.weight)
            if self.quant_conv.bias is not None:
                nn.init.zeros_(self.quant_conv.bias)

        # post_quant_conv: embed_dim -> z_channels
        if not try_identity_1x1(self.post_quant_conv):
            nn.init.xavier_uniform_(self.post_quant_conv.weight)
            if self.post_quant_conv.bias is not None:
                nn.init.zeros_(self.post_quant_conv.bias)

        # (Optional) zero the very last Conv2d inside decoder to keep outputs small at start
        if zero_final_decoder:
            last_conv = None
            for m in self.decoder.modules():
                if isinstance(m, nn.Conv2d):
                    last_conv = m
            if last_conv is not None:
                with torch.no_grad():
                    last_conv.weight.zero_()
                    if last_conv.bias is not None:
                        last_conv.bias.zero_()

    def encode(self, x):
        h = self.encoder(x)
        z = self.quant_conv(h)
        return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, mode='encode'):
        if mode == 'encode':
            return self.encode(x)
        elif mode == 'decode':
            return self.decode(x)
        elif mode == 'full':
            z = self.encode(x)
            x_hat = self.decode(z)
            return x_hat, z
        else:
            raise ValueError(f"Unknown mode {mode}")