import torch
import torch.nn.functional as F

def get_idx(z_flat, emb, use_znorm):
    """
    """
    if use_znorm:
        z_n = F.normalize(z_flat, dim=-1)                  # [N, C]
        e_nT = F.normalize(emb, dim=-1).t()                 # [C, V]
        idx = torch.argmax(z_n @ e_nT, dim=1)              # cosine max
    else:
        # L2: argmin ||z - e||^2 = argmin (||z||^2 + ||e||^2 - 2 z·e)
        z2 = (z_flat * z_flat).sum(dim=1, keepdim=True)    # [N, 1]
        e2 = (emb * emb).sum(dim=1)                        # [V]
        ze = z_flat @ emb.t()                              # [N, V]
        d = z2 + e2 - 2 * ze
        idx = torch.argmin(d, dim=1)
    return idx
