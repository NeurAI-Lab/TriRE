"""
Two is atleast as good as one!
Very similar to Born again neural netowrks (BNN)
"""

from masks.util import AbstractMask
from copy import deepcopy
import torch.nn as nn
from masks.util import re_init_weights
from torch.utils.data import TensorDataset


class TwoOne(AbstractMask):
    """
    Idea in the making
    """

    def __init__(self, args, device='cuda', backbone=None) -> None:
        super(TwoOne, self).__init__(args, device)
        self.device = device
        self.reset_backbone = None

    def create_second_net(self, backbone):
        self.reset_backbone = deepcopy(backbone)
        for layer in self.reset_backbone.parameters():
            new_weights = re_init_weights(layer.data.shape, self.device)
            layer.data = new_weights
        self.reset_backbone.to(self.device)

    def create_mask(self, backbone: nn.Module, dataset: TensorDataset = None) -> nn.Module:
        # create a random new model as a student
        self.create_second_net(backbone)

        # Exchange the parameters and train a new student
        net_params, reset_params = backbone.get_params(), self.reset_backbone.get_params()
        backbone.set_params(net_params)
        self.reset_backbone.set_params(net_params)

        for param in backbone.parameters():
            param.requires_grad = True

        return backbone
