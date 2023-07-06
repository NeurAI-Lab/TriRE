"""
Dense-Sparse-Dense training for deep neural networks
DSD
"""

import torch
import math
from masks.util import AbstractMask
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import TensorDataset

class DSD(AbstractMask):
    """
    Dense-Sparse-Dense training
    Computes the mask based on a threshold and zero out other weights during sparse training
    """
    def __init__(self, args, device='cuda', backbone=None) -> None:
        super(DSD, self).__init__(args, device)
        self.device = device
        self.dsd_sparsity = args.dsd_sparsity
        self.reset_backbone = deepcopy(backbone)
        self.create_dense_mask()

    def create_dense_mask(self):
        for param in self.reset_backbone.parameters():
            param.data[param.data == param.data] = 1
        self.reset_backbone.to(self.device)

    def create_sparse_mask(self, name, param, reset_param):
        reset_param.requires_grad = False
        N = torch.numel(param.data)
        k = math.floor(N * (1 - self.dsd_sparsity))
        if 'linear' in name:
            reset_param.data = torch.ones(reset_param.data.shape, device=self.device)
            return reset_param
        # TODO: Sort based only on the magnitude, not sign
        # TODO: set the unimportant parameters to zero
        sorted, indices = torch.sort(torch.flatten(param.data), descending=True)
        reset_param.data[param.data < sorted[k]] = 0
        reset_param.data[reset_param.data != 0] = 1
        return reset_param

    def create_mask(self, backbone: nn.Module, dataset: TensorDataset = None) -> nn.Module:
        self.reset_backbone = deepcopy(backbone)
        for (name, param), reset_param in zip(backbone.named_parameters(), self.reset_backbone.parameters()):
            reset_param = self.create_sparse_mask(name, param, reset_param)
            param.data = param.data * reset_param.data
        backbone.to(self.device)
        return backbone
