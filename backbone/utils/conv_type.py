"""
Borrowed from Knowledge Evolution in Neural Networks -- CVPR 2021 Oral
"""

import math
import torch
import numpy as np
import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd as autograd
# from configs.base_config import args as parser_args


DenseConv = nn.Conv2d


# Not learning weights, finding subnet
class SplitConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.split_mode = kwargs.pop('split_mode', None)
        self.split_rate = kwargs.pop('split_rate', None)
        self.in_channels_order = kwargs.pop('in_channels_order', None)
        # self.keep_rate = keep_rate
        super().__init__(*args, **kwargs)

        if self.split_mode == 'kels':
            if self.in_channels_order is None:
                mask = np.zeros((self.weight.size()))
                if self.weight.size()[1] == 3: ## This is the first conv
                    mask[:math.ceil(self.weight.size()[0] * self.split_rate), :, :, :] = 1
                else:
                    mask[:math.ceil(self.weight.size()[0] * self.split_rate), :math.ceil(self.weight.size()[1] * self.split_rate), :, :] = 1
            else:

                mask = np.zeros((self.weight.size()))
                conv_concat = [int(chs) for chs in self.in_channels_order.split(',')]
                # assert sum(conv_concat) == self.weight.size()[1],'In channels {} should be equal to sum(concat) {}'.format(self.weight.size()[1],conv_concat)
                start_ch = 0
                for conv in conv_concat:
                    mask[:math.ceil(self.weight.size()[0] * self.split_rate), start_ch:start_ch + math.ceil(conv * self.split_rate),
                    :, :] = 1
                    start_ch += conv

        elif self.split_mode == 'wels':
            mask = np.random.rand(*list(self.weight.shape))
            # threshold = np.percentile(scores, (1-self.keep_rate)*100)
            threshold = 1 - self.split_rate
            mask[mask < threshold] = 0
            mask[mask >= threshold] = 1
            if self.split_rate != 1:
                assert len(np.unique(mask)) == 2,'Something is wrong with the mask {}'.format(np.unique(mask))
        else:
            raise NotImplemented('Invalid split_mode {}'.format(self.split_mode))

        self.mask = nn.Parameter(torch.Tensor(mask), requires_grad=False)

    def extract_slim(self,dst_m,src_name,dst_name):
        c_out, c_in, _, _, = self.weight.size()
        d_out, d_in, _, _ = dst_m.weight.size()
        if self.in_channels_order is None:
            if c_in == 3:
                selected_convs = self.weight[:d_out]
                # is_first_conv = False
            else:
                selected_convs = self.weight[:d_out][:, :d_in, :, :]

            assert selected_convs.shape == dst_m.weight.shape
            dst_m.weight.data = selected_convs
        else:
            selected_convs = self.weight[:d_out, self.mask[0, :, 0, 0] == 1, :, :]
            assert selected_convs.shape == dst_m.weight.shape, '{} {} {} {}'.format(dst_name, src_name, dst_m.weight.shape,
                                                                                    selected_convs.shape)
            dst_m.weight.data = selected_convs
