import torch
import torch.nn as nn
import torch.nn.functional as F

class CutoffControlled(nn.Module):

    def __init__(self, base_loss_func=F.mse_loss):
        super().__init__()

        self.radius_cache = {}
        self.base_loss_func = base_loss_func

    def forward(self, hat, label, cutoff):
        """
        hat, label: (B, C, H, W)
        cutoff: float or tensor of shape (B,), each sample its own cutoff.
        """
        assert hat.shape == label.shape, "hat and label must have the same shape"
        B, C, H, W = label.shape
        assert cutoff.shape[0] == B, "cutoff must have length B"

        # clamp to (0, 1]
        cutoff = cutoff.clamp(1e-3, 1.0)

        # FFT over spatial dims
        hat_fft = torch.fft.fftn(hat, dim=(-2, -1))
        label_fft = torch.fft.fftn(label, dim=(-2, -1))

        if (H, W) in self.radius_cache:
            radius = self.radius_cache[(H, W)]
        else:
            # radius map (H, W)
            yy = torch.arange(H, device=label.device) - H // 2
            xx = torch.arange(W, device=label.device) - W // 2
            yy, xx = torch.meshgrid(yy, xx, indexing="ij")
            radius = torch.sqrt(yy**2 + xx**2)              # (H, W)

            r_max = radius.max()
            radius = radius.view(1, 1, H, W)                # (1, 1, H, W)
            self.radius_cache[(H, W)] = radius

        # per-sample cutoff: (B, 1, 1, 1)
        cutoff = (cutoff.view(B, 1, 1, 1) * r_max)

        # broadcast to (B, 1, H, W)
        mask = (radius <= cutoff).float()               # (B, 1, H, W)

        # shift mask to match unshifted FFT layout
        mask = torch.fft.ifftshift(mask, dim=(-2, -1))

        # apply low-pass filter
        hat_fft_lp = hat_fft * mask
        label_fft_lp = label_fft * mask

        # back to spatial domain (take real part)
        hat_lp = torch.fft.ifftn(hat_fft_lp, dim=(-2, -1)).real
        label_lp = torch.fft.ifftn(label_fft_lp, dim=(-2, -1)).real

        # MSE in low-frequency space
        loss = self.base_loss_func(hat_lp, label_lp)

        return loss

class CutoffFixed(nn.Module):

    def __init__(self, cutoff, base_loss_func=F.mse_loss):
        super().__init__()

        # clamp to (0, 1]
        cutoff = cutoff.clamp(1e-3, 1.0)

        self.cutoff = cutoff
        self.radius_cache = {}
        self.base_loss_func = base_loss_func

    def forward(self, hat, label):
        """
        hat, label: (B, C, H, W)
        cutoff: float or tensor of shape (B,), each sample its own cutoff.
        """
        assert hat.shape == label.shape, "hat and label must have the same shape"
        B, C, H, W = label.shape
        assert cutoff.shape[0] == B, "cutoff must have length B"


        # FFT over spatial dims
        hat_fft = torch.fft.fftn(hat, dim=(-2, -1))
        label_fft = torch.fft.fftn(label, dim=(-2, -1))

        if (H, W) in self.radius_cache:
            radius = self.radius_cache[(H, W)]
        else:
            # radius map (H, W)
            yy = torch.arange(H, device=label.device) - H // 2
            xx = torch.arange(W, device=label.device) - W // 2
            yy, xx = torch.meshgrid(yy, xx, indexing="ij")
            radius = torch.sqrt(yy**2 + xx**2)              # (H, W)

            r_max = radius.max()
            radius = radius.view(1, 1, H, W)                # (1, 1, H, W)
            self.radius_cache[(H, W)] = radius

        # per-sample cutoff: (B, 1, 1, 1)
        cutoff = (cutoff.view(B, 1, 1, 1) * r_max)

        # broadcast to (B, 1, H, W)
        mask = (radius <= cutoff).float()               # (B, 1, H, W)

        # shift mask to match unshifted FFT layout
        mask = torch.fft.ifftshift(mask, dim=(-2, -1))

        # apply low-pass filter
        hat_fft_lp = hat_fft * mask
        label_fft_lp = label_fft * mask

        # back to spatial domain (take real part)
        hat_lp = torch.fft.ifftn(hat_fft_lp, dim=(-2, -1)).real
        label_lp = torch.fft.ifftn(label_fft_lp, dim=(-2, -1)).real

        # MSE in low-frequency space
        loss = self.base_loss_func(hat_lp, label_lp)

        return loss
