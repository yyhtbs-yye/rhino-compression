import torch
import torch.nn as nn
import torch.nn.functional as F
from rhcore.utils.build_components import build_module

class BaseLossCombiner(nn.Module):

    def __init__(self, **named_configs):
        super().__init__()

        named_configs_copy = named_configs.copy()

        assert isinstance(named_configs_copy, dict), 'the arg ``named_configs_copy`` must be a dictionary'

        self.losses = nn.ModuleDict()
        self.loss_weight_schedulers = {}

        for name in named_configs_copy:
            config = named_configs_copy[name]

            if 'weight' in config:
                w = config.pop('weight')
                self.loss_weight_schedulers[name] = lambda _step: w
            elif 'weight_scheduler' in config:
                self.loss_weight_schedulers[name] = build_module(config.pop('weight_scheduler'))
            else:
                self.loss_weight_schedulers[name] = lambda _step: 1.0

            self.losses[name] = build_module(config)

    def forward(self, hat, label, step):

        total_loss = None
        
        for name in self.losses:
            if total_loss is None:
                total_loss = self.loss_weight_schedulers[name](step) * self.losses[name](hat, label)
            else:
                total_loss += self.loss_weight_schedulers[name](step) * self.losses[name](hat, label)

        return total_loss