"""
RIFLE: Backpropagation in depth for deep transfer learning through
re-initializing the fully connected layer
RIFLE
"""

import torch.nn as nn
from masks.util import AbstractMask
from masks.util import re_init_weights
from torch.utils.data import TensorDataset


class Rifle (AbstractMask):
    """
    Reset the final FC layer
    TODO: Cyclic learning rate (Refer the paper)
    """

    def __init__(self, args, device='cuda', backbone=None) -> None:
        super(Rifle, self).__init__(args, device)

    def create_mask(self, backbone: nn.Module, dataset:TensorDataset = None) -> nn.Module:
        new_weights = re_init_weights(backbone.linear.weight.shape, self.device)
        backbone.linear.weight.data = new_weights
        backbone.to(self.device)
        return backbone
