import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcompression.boats.sdvae_boat import SDVAEBoat

class VAVAEBoat(SDVAEBoat):
    def __init__(self, config=None):
        super().__init__(config=config or {})
        cfg = config or {}
        hps = cfg.get("boat", {}).get("hyperparameters", {})

        # VA-VAE specifics
        self.vf_weight      = float(hps.get('vf_weight', 1.0))       # whyper in the paper
        self.vf_margin_cos  = float(hps.get('vf_margin_cos', 0.1))   # m1
        self.vf_margin_dms  = float(hps.get('vf_margin_dms', 0.05))  # m2
        self.vf_eps         = float(hps.get('vf_eps', 1e-8))

        # Assert projector is defined
        assert 'proj' in config['boat']['models'], \
            "VA-VAE boat requires a 'projector' module in config['boat']['models']"

    # ---------- Core VF losses ----------
    def _vf_losses(self, z_feats, f_feats):
        """
        z_feats: (B, Cz, Hz, Wz) encoder last conv features
        f_feats: (B, Cf, Hf, Wf) VF spatial features (no grad)
        """
        # Project z to VF channel dim and align spatially
        z_proj = self.models['proj'](z_feats)                                  # (B, Cf, Hz, Wz)
        f_algn = F.interpolate(f_feats, size=z_proj.shape[-2:], mode='bilinear', align_corners=False)

        # L_mcos: per-location cosine, margin m1
        cos_loc = F.cosine_similarity(z_proj, f_algn, dim=1)                  # (B, Hz, Wz)
        l_mcos = F.relu(1.0 - self.vf_margin_cos - cos_loc).mean()

        # L_mdms: pairwise cosine matrices difference with margin m2
        B, Cf, H, W = z_proj.shape
        N = H * W

        z_flat = z_proj.flatten(2).transpose(1, 2)                             # (B, N, Cf)
        f_flat = f_algn.flatten(2).transpose(1, 2)                             # (B, N, Cf)

        z_norm = z_flat / (z_flat.norm(dim=-1, keepdim=True) + self.vf_eps)
        f_norm = f_flat / (f_flat.norm(dim=-1, keepdim=True) + self.vf_eps)

        sim_z = torch.matmul(z_norm, z_norm.transpose(1, 2))                  # (B, N, N)
        sim_f = torch.matmul(f_norm, f_norm.transpose(1, 2))                  # (B, N, N)

        l_mdms = F.relu((sim_z - sim_f).abs() - self.vf_margin_dms).mean()

        return l_mcos, l_mdms

    def _adaptive_weight(self, z, l_rec, l_vf_raw):
        """
        w_adaptive = ||∇_E l_rec|| / ||∇_E l_vf_raw||
        Gradients taken w.r.t. encoder last conv features z.
        """
        g_rec = torch.autograd.grad(l_rec, z, retain_graph=True, allow_unused=False)[0]
        g_vf  = torch.autograd.grad(l_vf_raw, z, retain_graph=True, allow_unused=False)[0]
        nr = g_rec.norm() + self.vf_eps
        nv = g_vf.norm()  + self.vf_eps
        return nr / nv

    # ---------- Train ---------- 
    # d_step_calc_losses is inherited from SDVAEBoat
    def g_step_calc_losses(self, batch):

        losses = super().g_step_calc_losses(batch)

        z_mean = batch['z_mean']
        x = batch['gt']

        # VF features (frozen)
        with torch.no_grad():
            f_feats = self.pretrained['vf'].get_features(x, batch['gt_path'])

        # VA-VAE raw terms
        l_mcos, l_mdms = self._vf_losses(z_mean, f_feats)
        l_vf = l_mcos + l_mdms

        g_loss = losses.pop('g_loss')

        w_adapt = self._adaptive_weight(z_mean, g_loss, l_vf)

        g_loss = (
            g_loss 
            + l_vf * self.vf_weight * w_adapt
        )

        return {
            'g_loss': g_loss,
            **losses,
            'l_mcos': l_mcos.detach(),
            'l_mdms': l_mdms.detach(),
            'l_vf': l_vf.detach(),
            'w_vf_adapt': w_adapt.detach(),
        }
