"""
The impact of reinitialization on generalization in convolutional neural networks
LayerWise (LW)
"""

import torch.nn as nn
from masks.util import AbstractMask
import numpy as np
from masks.util import re_init_weights
import re
import torch
from torch.utils.data import TensorDataset


class LayerWise(AbstractMask):
    """
    Incrementally mask earlier layer and insert a normalization layer
    Re-scale the masked layers while re-initialize the later layers.
    """

    def __init__(self, args, device='cuda', backbone=None) -> None:
        super(LayerWise, self).__init__(args, device)
        self.layer_threshold = args.lw_threshold
        self.mask_blocks = args.lw_mask_blocks
        self.record_scales(backbone)

    def record_scales(self, backbone):
        self.scales = {}
        for name, param in backbone.named_parameters():
            if 'bias' not in name:
                self.scales[name] = torch.linalg.norm(param.data)

    def create_mask(self, backbone: nn.Module, dataset: TensorDataset = None) -> nn.Module:
        for name, param in backbone.named_parameters():
            # Below code for re-initializing complete blocks
            if 'layer' in name:
                layer_num = int(re.search(r'\d+', name).group())
                if layer_num > self.layer_threshold:
                    # re-initialize
                    re_init_param = re_init_weights(param.data.shape, self.device)
                    param.data = re_init_param
                else:
                    # re-scale
                    scale = torch.linalg.norm(param.data)
                    print()

        # Increment the masked blocks
        self.layer_threshold += self.mask_blocks
        return backbone
