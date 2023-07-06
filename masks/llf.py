"""
Fortuitous Forgetting in Connectionist Networks
Later Layer Forgetting (LLF)
"""

import torch.nn as nn
from masks.util import AbstractMask
import re
from masks.util import re_init_weights
from torch.utils.data import TensorDataset


class LaterLayerForgetting(AbstractMask):
    """
    Mask layers equals or greater than layer_threshold
    Layers here correspond to resnet blocks
    """

    def __init__(self, args, device='cuda', backbone=None) -> None:
        super(LaterLayerForgetting, self).__init__(args, device)
        self.layer_threshold = args.llf_threshold

    def create_mask(self, backbone: nn.Module, dataset: TensorDataset = None) -> nn.Module:
        count = 0
        for name, param in backbone.named_parameters():
            if 'conv' in name:
                count += 1
                if count >= self.layer_threshold:
                    re_init_param = re_init_weights(param.data.shape, self.device)
                    param.data = re_init_param

                # Below code for re-initializing complete blocks
                # layer_num = int(re.search(r'\d+', name).group())
                # if layer_num >= self.layer_threshold:
                #     re_init_param = re_init_weights(param.data.shape, self.device)
                #     param.data = re_init_param
        return backbone
