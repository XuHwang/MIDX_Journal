import torch.nn as nn
from torch.nn.init import constant_

class normal_initialization(object):
    def __init__(self, initial_range=0.02) -> None:
        super().__init__()
        self.initial_range = initial_range

    def __call__(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initial_range)
            if module.padding_idx is not None:
                constant_(module.weight.data[module.padding_idx], 0.)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initial_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)