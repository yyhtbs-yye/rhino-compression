import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcompression.boats.sdvae_boat import SDVAEBoat

class DIVAEBoat(SDVAEBoat):

    def __init__(self, config=None):
        super().__init__(config=config or {})

    # ---------- G step with added dispersive term ----------
    def g_step_calc_losses(self, batch):

        losses = super().g_step_calc_losses(batch)

        results = {}

        results['w_disp'] = self.loss_weight_schedulers['dispersive_loss'](self.global_step())
        results['l_disp'] = self.losses['dispersive_loss'](self.out['z_sample'])

        g_loss = losses.pop('g_loss')

        # Total generator loss
        results['g_loss'] = (
            g_loss
            + results['l_disp'] * results['w_disp']
        )

        self.out = None

        return {k: v if k == 'g_loss' else torch.tensor(v).detach() for k, v in results.items()}
