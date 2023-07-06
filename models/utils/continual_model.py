import copy

import torch.nn as nn
from torch.optim import SGD, Adam
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from torch.nn import functional as F
from copy import deepcopy
import math
from masks.util import re_init_weights
from torch.utils.data import TensorDataset, DataLoader
import re
import numpy as np


def create_dense_mask_0(net, device, value):
    for param in net.parameters():
        param.data[param.data == param.data] = value
    net.to(device)
    return net


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []
    EXCLUDE_LAYERS_START_WITH = ['conv1', 'bn1', 'layer1', 'linear', 'classifier']
    EXCLUDE_LAYERS_CONTAINING = ['shortcut']

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.device = get_device()
        self.logsoft = nn.LogSoftmax(dim=1)
        self.fish = torch.zeros_like(self.net.get_params(), device=self.device)
        self.sparsity = args.sparsity
        self.net_init = deepcopy(self.net).to(self.device)
        self.net_mask_current = create_dense_mask_0(deepcopy(self.net), self.device, value=0)
        self.net_sparse_set = create_dense_mask_0(deepcopy(self.net), self.device, value=0)
        self.net_copy = create_dense_mask_0(deepcopy(self.net), self.device, value=1)
        self.net_grad = create_dense_mask_0(deepcopy(self.net), self.device, value=0)
        self.sparse_net_grad = create_dense_mask_0(deepcopy(self.net), self.device, value=0)    # for sparse model
        self.slow_lr_multiplier = args.slow_lr_multiplier
        self.lr_decay_rate = 0.999
        self.kwinner_mask = {}
        self.gradient_masks = {}
        self.net_epoch_k = create_dense_mask_0(deepcopy(self.net), self.device, value=1)    # rewind
        self.indices_1 = {}

    def measure_overlap(self):
        with torch.no_grad():
            N = 0
            total = 0
            for (name, mask_current), mask_set in \
                    zip(self.net_mask_current.named_parameters(), self.net_sparse_set.parameters()):
                if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                    continue
                elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                    continue
                total += torch.sum(torch.logical_and(mask_set.data, mask_set.data))
                N += torch.sum(torch.logical_and(mask_current.data, mask_set.data))
            overlap = N * 100 / total if total > 0 else 0
            print("{}% of weights from the cumulative sparse set"
                  " were reused in the current sparse set!".format(overlap))
        return overlap

    def measure_amount_of_sparsity(self):
        with torch.no_grad():
            N = 0
            total = 0
            for name, mask_set in self.net_sparse_set.named_parameters():
                total += torch.numel(mask_set.data)
                if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                    N += torch.numel(mask_set.data)
                elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                    N += torch.numel(mask_set.data)
                else:
                    N += torch.count_nonzero(mask_set.data)
            overlap = N * 100 / total if total > 0 else 0
            print("{}% of weights from the net"
                  " were reused in the sparse set!".format(overlap))
        return overlap

    def update_sparse_set(self):
        # add current important features to cumulative sparse set, overlapping ones will be overwritten
        for mask_current, mask_set in zip(self.net_mask_current.parameters(), self.net_sparse_set.parameters()):
            mask_set.data[mask_current.data == 1] = 1

    def mask_out_non_sparse_for_evaluation(self):
        # For evaluation, mask out the non-sparse set
        # If you want to retain rest of the parameters this step should be skipped
        for (name, param), mask_param in zip(self.net.named_parameters(), self.net_sparse_set.parameters()):
            if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                continue
            elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                continue
            param.data = param.data * mask_param.data

    def reparameterize_non_sparse(self):
        with torch.no_grad():
            # Re-initialize those params that were never part of sparse set
            if self.args.reinit_technique == 'init':
                for (name, param), mask_param, mask_current, param_init in \
                        zip(self.net.named_parameters(),
                            self.net_sparse_set.parameters(),
                            self.net_mask_current.parameters(),
                            self.net_init.parameters()):
                    if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                        continue
                    elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                        continue
                    mask = torch.zeros(mask_param.data.shape, device=self.device)
                    mask[mask_param.data == 1] = 1
                    mask[mask_current.data == 1] = 1
                    param.data = param.data * mask
                    param_init[mask == 1] = 0
                    param.data += param_init

            elif self.args.reinit_technique == 'rewind':    # Rewind non cumulative weights to epoch k from Retain phase
                for (name, param), mask_param, param_rewind in \
                        zip(self.net.named_parameters(),
                            self.net_sparse_set.parameters(),
                            self.net_epoch_k.parameters()):
                    if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                        continue
                    elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                        continue
                    mask = torch.zeros(mask_param.data.shape, device=self.device)
                    mask[mask_param.data == 1] = 1
                    param.data = param.data * mask
                    param_rewind[mask == 1] = 0
                    param.data += param_rewind

            else:
                for (name, param), mask_current, mask_param in zip(self.net.named_parameters(),
                                                     self.net_mask_current.parameters(),
                                                     self.net_sparse_set.parameters()):
                    if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                        continue
                    elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                        continue
                    mask = torch.zeros(mask_param.data.shape, device=self.device)
                    mask[mask_param.data == 1] = 1
                    mask[mask_current.data == 1] = 1
                    param.data = param.data * mask
                    re_init_param = re_init_weights(param.data.clone(), param.data.shape, self.device,
                                                    self.args.reinit_technique)
                    re_init_param[mask == 1] = 0
                    param.data = param.data + re_init_param

    def extract_new_sparse_model(self, t) -> nn.Module:
        # Re-init the current task mask
        self.net_mask_current = create_dense_mask_0(deepcopy(self.net), self.device, value=0)
        w_grad = None
        # Extract sparse model for the current task
        start_idx = 0
        with torch.no_grad():
            if self.args.pruning_technique in ['fisher_pruning', 'magnitude_pruning', 'CWI']:
                for (name, param), param_mask in \
                        zip(self.net.named_parameters(),
                            self.net_mask_current.parameters()):
                    if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                        start_idx += torch.numel(param.data)
                        continue
                    elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                        start_idx += torch.numel(param.data)
                        continue
                    N = param.data.shape[0]
                    k = math.floor(N * self.args.kwinner_sparsity)
                    shape = param.shape
                    # print(name, param.data)
                    if self.args.pruning_technique == 'fisher_pruning':
                        end_idx = start_idx + torch.numel(param.data)
                        param_fish = self.fish[start_idx: end_idx]
                        param_fish = param_fish.reshape(param.shape)
                        start_idx += torch.numel(param.data)
                        adjusted_importance = param_fish.clone()
                    elif self.args.pruning_technique == 'magnitude_pruning':
                        adjusted_importance = param.data.clone()
                    elif self.args.pruning_technique == 'CWI':  # Continuous weight importance
                        weight = param.data.detach().clone().cpu().numpy()
                        if not param.grad is None:
                            grad_copy = copy.deepcopy(param.grad)
                            w_grad = grad_copy.detach().clone().cpu().numpy()
                        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
                        if not w_grad is None:
                            grad_temp = np.abs(w_grad)
                            adjusted_importance = weight_temp + grad_temp
                            adjusted_importance = torch.from_numpy(adjusted_importance).to(self.device)
                        else:
                            adjusted_importance = weight_temp
                            adjusted_importance = torch.from_numpy(adjusted_importance).to(self.device)
                    self.structured_pruning(k, name, shape, param_mask, adjusted_importance, t)
            elif self.args.pruning_technique == 'init_pruning':
                for (name, param), param_mask, param_init in \
                        zip(self.net.named_parameters(),
                            self.net_mask_current.parameters(),
                            self.net_init.parameters()):
                    if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                        continue
                    elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                        continue
                    N = param.data.shape[0]
                    k = math.floor(N * self.args.kwinner_sparsity)
                    shape = param.shape
                    adjusted_importance = torch.abs(param.data - param_init.data)
                    self.structured_pruning(k, name, shape, param_mask, adjusted_importance, t)

    def structured_pruning(self, k, name, shape, param_mask, adjusted_importance, t):
        # Use k-winner take all mask first
        tmp = [int(s) for s in re.findall(r'\d+', name)]
        # FixMe: Is there a better way to handle this?
        if self.args.dataset == 'perm-mnist':   # for DIL scenario
            mask_name = "self.net.kwinner{}.act_count".format(tmp[0])
        else:
            mask_name = "self.net.layer{}[{}].kwinner{}.act_count".format(tmp[0], tmp[1], tmp[2])

        kwinner_mask = eval(mask_name)

        if self.args.use_het_drop:
            if tmp[0] == 2:
                het_drop = self.args.het_drop
            elif tmp[0] == 3:
                het_drop = self.args.het_drop * 1.5
            else:
                het_drop = self.args.het_drop * 3.0

            prob_drop = torch.zeros(kwinner_mask.shape, device=self.device)
            for i, element in enumerate(kwinner_mask):
                norm = -(element / torch.max(kwinner_mask))
                prob_drop[i] = torch.exp(norm * het_drop)
            indices_1 = torch.where(prob_drop > torch.mean(prob_drop))[0]

        if self.args.use_kwta:
            if self.args.non_overlaping_kwinner and t > 0:
                cumulative_mask = self.kwinner_mask[(0, mask_name)]
                for i in range(1, t):
                    cumulative_mask[self.kwinner_mask[(i, mask_name)] == 1] = 1
                kwinner_mask[cumulative_mask == 1] = 0
            act_counts, indices_1 = torch.topk(kwinner_mask, k)

        mask = torch.empty(shape[0], device=self.device).fill_(float('-inf'))
        mask.scatter_(0, indices_1, 1)

        # Log the mask
        self.kwinner_mask[(t, mask_name)] = mask

        # Prune the layer based on k-winner mask
        num_filters = shape[0]
        if 'conv' in name and len(shape) > 1:
            for filter_idx in range(num_filters):
                if mask[filter_idx] < 0:
                    adjusted_importance[filter_idx, :, :, :] = mask[filter_idx]

            N = (mask == 1).sum() * torch.numel(adjusted_importance[0, :, :, :])
            l = math.floor(N * self.sparsity)
            indices_2 = torch.topk(torch.flatten(adjusted_importance), l)[1]

            pruning_mask = torch.zeros(torch.numel(adjusted_importance), device=self.device)
            pruning_mask.scatter_(0, indices_2, 1)
            param_mask.data += pruning_mask.reshape(shape)
            self.gradient_masks[name] = param_mask.data.cuda()
        elif 'fc' in name and len(shape) > 1:                       # For DIL scenario
                N = (mask == 1).sum() * torch.numel(adjusted_importance[0, :])
                l = math.floor(N * self.sparsity)
                indices_2 = torch.topk(torch.flatten(adjusted_importance), l)[1]

                pruning_mask = torch.zeros(torch.numel(adjusted_importance), device=self.device)
                pruning_mask.scatter_(0, indices_2, 1)
                param_mask.data += pruning_mask.reshape(shape)
                self.gradient_masks[name] = param_mask.data.cuda()
        else:
            adjusted_importance[mask < 0] = float('-inf')
            N = (mask == 1).sum()
            l = math.floor(N * self.sparsity)
            indices_2 = torch.topk(torch.flatten(adjusted_importance), l)[1]
            param_mask.data[indices_2] = 1

    def compute_fisher(self, dataset):
        fish = torch.zeros_like(self.net.get_params())
        inputs_, labels_ = self.buffer.get_all_data(transform=self.transform)
        sample_dataset = TensorDataset(inputs_, labels_)
        samples_loader = DataLoader(sample_dataset, batch_size=self.args.minibatch_size)
        for j, data in enumerate(samples_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.net_copy.zero_grad()
                output = self.net_copy(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net_copy.get_grads() ** 2

        fish /= (len(dataset.train_loader) * self.args.batch_size)

        self.fish *= self.args.gamma
        self.fish += fish

    def prune_apply_masks_on_grads_efficient(self):
        with torch.no_grad():
            for name, W in (self.net.named_parameters()):
                if name in self.gradient_masks:
                    dtype = W.dtype
                    # print("before", W.grad)
                    (W.grad).mul_((self.gradient_masks[name] != 0).type(dtype))
                    # print("after", W.grad)
                    pass

    def update_non_cumulative_sparse(self, observe=False):
        with torch.no_grad():
            # update weights NOT in cumulative sparse set. Those in sparse set are not updated.
            for (name, param_net), param_sparse, param_grad_copy in \
                    zip(self.net.named_parameters(),
                        self.net_sparse_set.parameters(),
                        self.net_grad.parameters()):
                param_lr = torch.ones(param_sparse.data.shape, device=self.device)
                # Exclude final linear layer weights from masking
                if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                    param_grad_copy.grad = param_net.grad.clone()
                    continue
                elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                    param_grad_copy.grad = param_net.grad.clone()
                    continue
                param_lr[param_sparse == 1] = 0
                param_grad_copy.grad = param_net.grad.clone() * param_lr
                # For rewind phase
                if observe:
                    param_net.grad += param_grad_copy.grad.clone()

    def update_cumulative_sparse(self):
        with torch.no_grad():
            # update weights in cumulative sparse set. Those NOT in sparse set are not updated.
            for (name, param_net), param_sparse, param_grad_copy in \
                    zip(self.net.named_parameters(),
                        self.net_sparse_set.parameters(),
                        self.net_grad.parameters()):
                # Exclude final linear layer weights from masking
                if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                    param_net.grad += param_grad_copy.grad.clone()
                    continue
                elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                    param_net.grad += param_grad_copy.grad.clone()
                    continue
                param_net.grad *= param_sparse
                param_net.grad += param_grad_copy.grad.clone()

    def update_non_overlapping_sparse(self, is_sparse_m=False):
        with torch.no_grad():
            if is_sparse_m:
                for (name, param_net), param_current, param_sparse, param_grad_copy in \
                        zip(self.sparse_model.named_parameters(),
                            self.net_mask_current.parameters(),
                            self.net_sparse_set.parameters(),
                            self.sparse_net_grad.parameters()):
                    param_lr = torch.ones(param_sparse.data.shape, device=self.device)
                    # Exclude final linear layer weights from masking
                    if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                        param_grad_copy.grad = param_net.grad.clone()
                        continue
                    elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                        param_grad_copy.grad = param_net.grad.clone()
                        continue
                    # no update using overlapping weights' gradients
                    param_lr[param_current == param_sparse] = 0
                    # No gradient for weights that are not part of current sparse mask
                    param_lr[param_current == 0] = 0
                    param_grad_copy.grad = param_net.grad.clone() * param_lr

            else:
                # Update gradients of current sparse mask that are NOT in sparse set
                for (name, param_net), param_current, param_sparse, param_grad_copy in \
                        zip(self.net.named_parameters(),
                            self.net_mask_current.parameters(),
                            self.net_sparse_set.parameters(),
                            self.net_grad.parameters()):
                    param_lr = torch.ones(param_sparse.data.shape, device=self.device)
                    # Exclude final linear layer weights from masking
                    if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                        param_grad_copy.grad = param_net.grad.clone()
                        continue
                    elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                        param_grad_copy.grad = param_net.grad.clone()
                        continue
                    # no update using overlapping weights' gradients
                    param_lr[param_current == param_sparse] = 0
                    # No gradient for weights that are not part of current sparse mask
                    param_lr[param_current == 0] = 0
                    param_grad_copy.grad = param_net.grad.clone() * param_lr

    def update_overlapping_sparse(self, is_sparse_m=False):
        with torch.no_grad():
            if is_sparse_m:
                for (name, param_net), param_current, param_sparse, param_grad_copy in \
                        zip(self.sparse_model.named_parameters(),
                            self.net_mask_current.parameters(),
                            self.net_sparse_set.parameters(),
                            self.sparse_net_grad.parameters()):
                    # Exclude some layers from masking
                    if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                        param_net.grad += param_grad_copy.grad.clone()
                        continue
                    elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                        param_net.grad += param_grad_copy.grad.clone()
                        continue
                    param_net.grad = param_net.grad * param_sparse  # all f_\theta \in S
                    param_net.grad += param_grad_copy.grad.clone()

            else:
                # Update gradients of current sparse mask that are in sparse set
                for (name, param_net), param_current, param_sparse, param_grad_copy in \
                        zip(self.net.named_parameters(),
                            self.net_mask_current.parameters(),
                            self.net_sparse_set.parameters(),
                            self.net_grad.parameters()):
                    # Exclude some layers from masking
                    if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                        param_net.grad += param_grad_copy.grad.clone()
                        continue
                    elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                        param_net.grad += param_grad_copy.grad.clone()
                        continue
                    param_net.grad = param_net.grad * param_sparse  # all f_\theta \in S
                    param_net.grad += param_grad_copy.grad.clone()

    def reload_non_overlapping_cum_sparse(self):
        # Reload the weights of the non-overlapping cumulative sparse set alone.
        # FixMe: This fn does not load plastic weights at the moment
        for (name, param_net), param_current, param_sparse, param_copy in \
                zip(self.net.named_parameters(),
                    self.net_mask_current.parameters(),
                    self.net_sparse_set.parameters(),
                    self.net_copy.parameters()):
            if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                continue
            elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                continue
            # take out only non-overlapping cumulative sparse from saved copy
            param_copy.data = param_copy.data * param_sparse
            param_copy.data[param_current == 1] = 0
            # Retain only current sparse in the net.
            # param_net.data *= param_current
            # add these disjoint sets
            param_net.data += param_copy.data

    def maskout_non_sparse(self, is_sparse_m=False):
        # Mask out weights not in the current sparse mask
        if is_sparse_m:
            for (name, param_net), param_current, param_sparse in zip(self.sparse_model.named_parameters(),
                                                                      self.net_mask_current.parameters(),
                                                                      self.net_sparse_set.parameters()):
                if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                    continue
                elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                    continue
                mask = torch.zeros(param_current.data.shape, device=self.device)
                mask[param_current.data == 1] = 1
                mask[param_sparse.data == 1] = 1
                param_net.data = param_net.data * mask

        else:
            for (name, param_net), param_current, param_sparse in zip(self.net.named_parameters(),
                                                       self.net_mask_current.parameters(),
                                                       self.net_sparse_set.parameters()):
                if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                    continue
                elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                    continue
                mask = torch.zeros(param_current.data.shape, device=self.device)
                mask[param_current.data == 1] = 1
                mask[param_sparse.data == 1] = 1
                param_net.data = param_net.data * mask

    def maskout_cum_sparse(self):
        # Mask out weights not in the current sparse mask
        for (name, param_net), param_current, param_sparse in zip(self.net.named_parameters(),
                                                   self.net_mask_current.parameters(),
                                                   self.net_sparse_set.parameters()):
            if any(name.startswith(layer) for layer in ContinualModel.EXCLUDE_LAYERS_START_WITH):
                continue
            elif any(layer in name for layer in ContinualModel.EXCLUDE_LAYERS_CONTAINING):
                continue
            mask = torch.ones(param_current.data.shape, device=self.device)
            mask[param_current.data == 1] = 0
            mask[param_sparse.data == 1] = 0
            param_net.data = param_net.data * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass