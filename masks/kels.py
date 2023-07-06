"""
Knowledge Evolution in neural networks
KELS
"""

import torch
from masks.util import AbstractMask
from copy import deepcopy
import numpy as np
import torch.nn as nn
from masks.util import re_init_weights
import math
from torch.utils.data import TensorDataset

class KELS(AbstractMask):
    """
    Initialize a kernel-level structured mask using a split_rate and create fit & reset hypothesis.
    """
    def __init__(self, args, device='cuda', backbone=None) -> None:
        super(KELS, self).__init__(args, device)
        self.device = device
        self.layer_threshold = args.llf_threshold
        self.split_rate = args.kels_split_rate
        self.reset_backbone = None

    def create_reset_weights(self):
        for name, reset_param in self.reset_backbone.named_parameters():
            reset_param.requires_grad = False
            exclude_lst = ['linear', 'bn', 'bias']
            if any(layer in name for layer in exclude_lst):
                mask = torch.ones(reset_param.data.shape)
                reset_param.data = torch.Tensor(mask)
                continue
            elif 'conv' in name:
                mask = np.zeros((reset_param.data.size()))
                if reset_param.data.size()[1] == 3:
                    mask[:math.ceil(reset_param.data.size()[0] * self.split_rate), :, :, :] = 1
                else:
                    mask[:math.ceil(reset_param.data.size()[0] * self.split_rate),
                    :math.ceil(reset_param.data.size()[1] * self.split_rate), :, :] = 1
            else:
                mask = np.random.rand(*list(reset_param.data.shape))
                mask[mask < self.split_rate] = 0
                mask[mask >= self.split_rate] = 1

            if self.split_rate != 1 and 'linear' not in name:
                assert len(np.unique(mask)) == 2, 'Something is wrong with the mask {}'.format(np.unique(self.mask))
            reset_param.data = torch.Tensor(mask)
        self.reset_backbone.to(self.device)

    def create_mask(self, backbone: nn.Module, dataset: TensorDataset = None) -> nn.Module:
        if self.reset_backbone is None:
            # Mask is initialized only once during CL training
            self.reset_backbone = deepcopy(backbone)
            self.create_reset_weights()
        for (name, param), reset_param in zip(backbone.named_parameters(), self.reset_backbone.parameters()):
            re_init_param = re_init_weights(param.data.shape, self.device)
            param.data = param.data * reset_param.data
            re_init_param[reset_param == 1] = 0
            param.data = param.data + re_init_param
        return backbone
