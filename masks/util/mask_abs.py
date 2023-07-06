from abc import abstractmethod

import torch.nn as nn
import torch
from argparse import Namespace
from torch.utils.data import TensorDataset


class AbstractMask(nn.Module):
    """
    Mask subset of backbone.
    Final Linear layer is never masked except in RIFLE
    """
    def __init__(self, args: Namespace, device) -> None:
        super(AbstractMask, self).__init__()
        self.args = args
        self.device = device

    def forward(self, backbone: nn.Module, dataset: TensorDataset = None) -> nn.Module:
        """
        Returns the masked backbone
        """
        return self.create_mask(backbone, dataset).to(self.device)

    @abstractmethod
    def create_mask(self, backbone: nn.Module, dataset: TensorDataset = None) -> nn.Module:
        """
        Computes a mask and applies it on the backbone.
        The masked parts are re-initialized using xavier initialization
        """
        pass

