"""
Retain and Revise Randomly using Fisher Information Matrix
The selection of top-k parameters is random
"""
import torch.nn as nn
from masks.util import AbstractMask
import re
from masks.util import re_init_weights
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math
from copy import deepcopy


class RRREMA(AbstractMask):
    """
    Mask layers equals or greater than layer_threshold
    Layers here correspond to resnet blocks
    """

    def __init__(self, args, device='cuda', backbone=None) -> None:
        super(RRREMA, self).__init__(args, device)
        self.retain_rate = args.rrr_retain_rate
        self.gamma = args.rrr_fisher_ema_gamma
        self.fish = None
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.reset_backbone = deepcopy(backbone)
        self.create_dense_mask()

    def compute_fisher(self, backbone, dataset):
        fish = torch.zeros_like(backbone.get_params())
        train_loader = DataLoader(dataset, batch_size=self.args.minibatch_size)
        for j, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):

                output = backbone(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * backbone.get_grads() ** 2

        fish /= (len(train_loader) * self.args.batch_size)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish = self.gamma * self.fish + (1 - self.gamma) * fish
            # self.fish += fish

        # increase retain rate
        self.retain_rate += self.retain_rate * 0.005

    def create_dense_mask(self):
        for param in self.reset_backbone.parameters():
            param.data[param.data == param.data] = 1
        self.reset_backbone.to(self.device)

    def create_sparse_mask(self, name, param, reset_param, param_fish):
        reset_param.requires_grad = False
        # select all the weights
        reset_param.data[reset_param.data == reset_param.data] = 1
        N = torch.numel(param.data)
        k = math.floor(N * self.retain_rate)
        exclude_lst = ['linear', 'bn', 'bias']
        if  any(layer in name for layer in exclude_lst):
            reset_param.data = torch.ones(reset_param.data.shape, device=self.device)
            return reset_param
        sorted, indices = torch.sort(torch.flatten(param_fish), descending=True)
        # de-select based Fisher score
        reset_param.data[param_fish < sorted[k]] = 0

    def create_mask(self, backbone: nn.Module, dataset: TensorDataset = None) -> nn.Module:
        if dataset is not None:
            self.compute_fisher(backbone, dataset)
            self.create_dense_mask()
            start_idx = 0
            for (name, param), reset_param in zip(backbone.named_parameters(), self.reset_backbone.parameters()):
                n_params = torch.numel(param.data)
                end_idx = start_idx + n_params
                param_fish = self.fish[start_idx: end_idx]
                param_fish = param_fish.reshape(param.shape)

                self.create_sparse_mask(name, param, reset_param, param_fish)

                # re-init part of the params
                re_init_param = re_init_weights(param.data.shape, self.device)
                param.data = param.data * reset_param.data
                re_init_param[reset_param == 1] = 0
                param.data = param.data + re_init_param

                start_idx += n_params
        else:
            raise NotImplementedError("Provide a TensorDataset to compute Fisher matrix!")
        return backbone
