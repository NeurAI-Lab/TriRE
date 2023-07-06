"""
On warm-starting neural networks
Shrink, Perturb , Repeat (SPR)
"""

import torch.nn as nn
import torch
from masks.util import AbstractMask
import numpy as np
from copy import deepcopy
from utils.buffer import Buffer
from torch.utils.data import TensorDataset


class ShrinkPerturbRepeat (AbstractMask):
    """
    Shrink the layers using scaling factor lambda and perturb using gaussian noise
    lambda=0 and sigma=1 works the best for the primitive incremental learning presented in the paper.
    """

    def __init__(self, args, device='cuda', backbone=None) -> None:
        super(ShrinkPerturbRepeat, self).__init__(args, device)
        self.lambda_scale = args.spr_lambda_scale
        self.noise_scale = args.spr_noise_scale

    def create_mask(self, backbone: nn.Module, dataset: TensorDataset = None) -> nn.Module:
        linear_weights = deepcopy(backbone.linear.weight.data)
        noise = np.random.normal(loc=0, scale=self.noise_scale, size=backbone.get_params().shape).astype(np.float32)

        # Shrink using lambda and add perturbation
        backbone.set_params(self.lambda_scale * backbone.get_params() + torch.tensor(noise).to(self.device))

        # retain final linear layer as is
        backbone.linear.weight.data = linear_weights

        return backbone
