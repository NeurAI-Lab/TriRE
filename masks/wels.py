"""
Knowledge Evolution in neural networks
WELS
"""

import torch
from masks.util import AbstractMask
import numpy as np
import torch.nn as nn
from masks.util import re_init_weights
from copy import deepcopy
from torch.utils.data import TensorDataset


class WELS(AbstractMask):
    """
    Initialize a mask randomly using a split_rate and create fit & reset hypothesis.
    """
    def __init__(self, args, device='cuda', backbone=None) -> None:
        super(WELS, self).__init__(args, device)
        self.device = device
        self.split_rate = args.wels_split_rate
        self.reset_backbone = None

    def create_reset_weights(self):
        for name, reset_param in self.reset_backbone.named_parameters():
            reset_param.requires_grad = False
            exclude_lst = ['linear', 'bn', 'bias']
            if any(layer in name for layer in exclude_lst):
                mask = torch.ones(reset_param.data.shape)
                reset_param.data = torch.Tensor(mask)
                continue
            mask = np.random.rand(*list(reset_param.data.shape))
            mask[mask < self.split_rate] = 0
            mask[mask >= self.split_rate] = 1
            if self.split_rate != 1:
                assert len(np.unique(mask)) == 2, 'Something is wrong with the mask {}'.format(np.unique(self.mask))
            reset_param.data = torch.Tensor(mask)
        self.reset_backbone.to(self.device)

    def create_mask(self, backbone: nn.Module, dataset:TensorDataset = None) -> nn.Module:
        if self.reset_backbone is None:
            # Mask is initialized only once during CL training
            self.reset_backbone = deepcopy(backbone)
            self.create_reset_weights()
        for (name, param), reset_param in zip(backbone.named_parameters(), self.reset_backbone.parameters()):
            re_init_param = re_init_weights(param.data.shape, self.device, reinint_method='kaiming')
            param.data = param.data * reset_param.data
            re_init_param[reset_param == 1] = 0
            param.data = param.data + re_init_param
        return backbone
