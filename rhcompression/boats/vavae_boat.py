import torch
import torch.nn.functional as F
from rhtrain.utils.ddp_utils import move_to_device
from rhcompression.boats.sdvae_boat import SDVAEBoat

class VAVAEBoat(SDVAEBoat):
    """
    VA-VAE boat:
      L_vf = w_hyper * w_adaptive * (L_mcos + L_mdms)
      w_adaptive = ||∇_E L_rec|| / ||∇_E (L_mcos + L_mdms)||   (E: last encoder conv features)
      L_mcos  = mean_{i,j} ReLU(1 - m1 - cos(z'_{ij}, f_{ij}))
      L_mdms  = mean_{i,j} ReLU(|cos(z_i, z_j) - cos(f_i, f_j)| - m2)

    Expects:
      self.models['net']        -> returns (x_hat, q, enc_feats)
      self.models['proj']       -> projector mapping enc_feats -> VF channel dim
      self.pretrained['vf']     -> frozen VF wrapper with .get_features(x)
    """
    def __init__(self, config=None):
        super().__init__(config=config or {})
        hps = config['boat'].get('hyperparameters', {})

        # VA-VAE specifics
        self.vf_weight      = float(hps.get('vf_weight', 1.0))       # whyper in the paper
        self.vf_margin_cos  = float(hps.get('vf_margin_cos', 0.1))   # m1
        self.vf_margin_dms  = float(hps.get('vf_margin_dms', 0.05))  # m2
        self.vf_eps         = float(hps.get('vf_eps', 1e-8))

        # Assert projector is defined
        assert 'proj' in config['boat']['models'], \
            "VA-VAE boat requires a 'projector' module in config['boat']['models']"

    # ---------- Core VF losses ----------
    def _vf_losses(self, z_feat, f_feat):
        """
        z_feat: (B, Cz, Hz, Wz) encoder last conv features
        f_feat: (B, Cf, Hf, Wf) VF spatial features (no grad)
        """
        # Project z to VF channel dim and align spatially
        z_proj = self.models['proj'](z_feat)                                  # (B, Cf, Hz, Wz)
        f_algn = F.interpolate(f_feat, size=z_proj.shape[-2:], mode='bilinear', align_corners=False)

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

    def _adaptive_weight(self, z_feat, l_rec, l_vf_raw):
        """
        w_adaptive = ||∇_E l_rec|| / ||∇_E l_vf_raw||
        Gradients taken w.r.t. encoder last conv features z_feat.
        """
        g_rec = torch.autograd.grad(l_rec, z_feat, retain_graph=True, allow_unused=False)[0]
        g_vf  = torch.autograd.grad(l_vf_raw, z_feat, retain_graph=True, allow_unused=False)[0]
        nr = g_rec.norm() + self.vf_eps
        nv = g_vf.norm()  + self.vf_eps
        return nr / nv

    # ---------- Train ---------- 
    # d_step_calc_losses is inherited from SDVAEBoat
    def g_step_calc_losses(self, batch):

        batch = move_to_device(batch, self.device)
        
        x = batch['gt']

        x_hat, z_feat, q = self.models['net'](x, mode='full', sample_method='random')

        l_img = self.losses['pixel_loss'](x_hat, x)

        w_adv = self.max_weight_adv * float(self.adv_fadein(self.global_step()))
        if self.global_step() >= self.start_adv:
            d_fake_for_g = self.models['critic'](x_hat)
            train_out = {'real': d_fake_for_g, 'fake': None, **batch}
            l_adv = self.losses['critic'](train_out)
        else:
            l_adv = torch.zeros((), device=self.device)

        w_lpips = self.max_weight_lpips * float(self.lpips_fadein(self.global_step()))
        l_lpips = (self.losses['lpips_loss'](x_hat, x).mean()
                   if ('lpips_loss' in self.losses and w_lpips > 1e-6)
                   else torch.zeros((), device=self.device))

        l_kl = (q.kl().mean() if hasattr(q, 'kl') else torch.zeros((), device=self.device))

        # VF features (frozen)
        with torch.no_grad():
            f_feat = self.pretrained['vf'].get_features(x, batch['gt_path'])

        # VA-VAE raw terms
        l_mcos, l_mdms = self._vf_losses(z_feat, f_feat)
        l_vf_raw = l_mcos + l_mdms

        # Adaptive weighting (computed before adding vf_weight)
        l_rec_for_grad = self.lambda_image * l_img + w_adv * l_adv + w_lpips * l_lpips + self.beta_kl * l_kl
        w_adapt = self._adaptive_weight(z_feat, l_rec_for_grad, l_vf_raw)

        l_vf = self.vf_weight * w_adapt * l_vf_raw

        g_loss = l_rec_for_grad + l_vf

        return {
            'g_loss': g_loss,
            'l_image': l_img.detach(),
            'w_adv': torch.tensor(w_adv),
            'l_adv': l_adv.detach(),
            'w_lpips': torch.tensor(w_lpips, device=self.device),
            'l_lpips': l_lpips.detach(),
            'l_kl': l_kl.detach(),
            'l_mcos': l_mcos.detach(),
            'l_mdms': l_mdms.detach(),
            'l_vf': l_vf_raw.detach(),
            'w_vf_adapt': w_adapt.detach(),
        }
