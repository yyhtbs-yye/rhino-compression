import torch

def rbf_kernel(x, y, sigma):
    """
    x: (B, D), y: (B, D)
    returns K in R^{B x B}
    """
    x = x.unsqueeze(1)  # (B, 1, D)
    y = y.unsqueeze(0)  # (1, B, D)
    diff = x - y
    dist_sq = (diff ** 2).sum(-1)  # (B, B)
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))
