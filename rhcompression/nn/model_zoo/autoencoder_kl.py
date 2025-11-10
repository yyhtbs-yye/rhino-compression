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

    def forward(self, x, mode='encode', sample_method=None, noise_weight=None):

        if mode == 'encode' and sample_method is None:
            return {
                'q': self.encode(x)               # return posterior only
            }
        elif mode == 'encode' and sample_method == 'random':
            posterior = self.encode(x)
            z, e = posterior.sample()
            return {
                'z_sample': z, 
                'e_sample': e
            }
        elif mode == 'encode' and sample_method == 'mode':
            posterior = self.encode(x)
            z = posterior.mode()
            return {
                'z_mean': z
            }
        elif mode == 'full':
            posterior = self.encode(x)
            z_mean = posterior.mode()
            z_sample, e_sample = posterior.sample()
            x_mean = self.decode(z_mean)
            if noise_weight is None:
                x_sample = self.decode(z_sample)
                x_sample_sg = self.decode(z_sample.detach())
            else:
                x_sample = self.decode(z_sample)
                x_sample_sg = self.decode((z_mean + e_sample * noise_weight).detach())
            return {'x_mean': x_mean, 
                    'x_sample': x_sample, 
                    'x_sample_sg': x_sample_sg, 
                    'z_mean': z_mean, 
                    'z_sample': z_sample,
                    'e_sample': e_sample, 
                    'q': posterior}

        elif mode == 'decode':
            return self.decode(x)
        else:
            raise ValueError(f"Unknown mode {mode}")
