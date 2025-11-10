import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEMA(nn.Module):
    """
    Keeps an EMA of the state and returns alpha so that
    exp(-alpha * E[state]) ≈ target_mag.
    """
    def __init__(self, init=1.0, momentum=0.99, target_mag=0.3):
        super().__init__()
        self.momentum = momentum
        self.target_mag = float(target_mag)
        # buffer so it’s saved in state_dict, can .to(device), etc.
        self.register_buffer("value", torch.tensor(float(init)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B,) tensor of states
        returns: scalar alpha tensor
        """
        with torch.no_grad():
            batch_mean = state.mean()
            if self.training:  # only update EMA in training mode
                self.value.mul_(self.momentum).add_(batch_mean * (1 - self.momentum))

            dist_mean = self.value.clamp(min=1e-6).to(state.device, state.dtype)
            alpha = -torch.log(torch.tensor(self.target_mag,
                                            device=state.device,
                                            dtype=state.dtype)) / dist_mean
            alpha = alpha.clamp(1e-3, 1e3)
        return alpha