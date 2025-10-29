import torch
import torch.nn as nn
from rhcompression.nn.blocks.compvis.encdec import Encoder, Decoder
from rhcompression.nn.distributions.diagnal_gaussian import DiagonalGaussianDistribution

class AutoencoderKL(nn.Module):
    def __init__(self, ddconfig, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"] == True
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, mode='encode', sample_method=None):

        if mode == 'encode' and sample_method is None:
            return self.encode(x)               # return posterior only
        elif mode == 'encode' and sample_method == 'random':
            posterior = self.encode(x)
            z = posterior.sample()
            return z
        elif mode == 'encode' and sample_method == 'mode':
            posterior = self.encode(x)
            z = posterior.mode()
            return z
        elif mode == 'full' and sample_method == 'random':
            posterior = self.encode(x)
            z = posterior.sample()
            x_hat = self.decode(z)
            return x_hat, posterior
        elif mode == 'full' and sample_method == 'mode':
            posterior = self.encode(x)
            z = posterior.mode()
            x_hat = self.decode(z)
            return x_hat, posterior
        elif mode == 'decode':
            return self.decode(x)
        else:
            raise ValueError(f"Unknown mode {mode}")
