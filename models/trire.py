import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.buffer_tricks import Buffer
from copy import deepcopy
from torch import nn


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning retain and revise method with Dense-Sparse-Dense.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--gamma', type=float, required=True,
                        help='gamma parameter for EWC online')
    parser.add_argument('--het_drop', type=float, default=1, help='heterogenous dropout weight')
    return parser


class TriRE(ContinualModel):
    NAME = 'trire'
    COMPATIBILITY = ['class-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(TriRE, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.ema_model = deepcopy(self.net).to(self.device)
        self.reg_weight = args.reg_weight
        self.ema_update_freq = args.stable_model_update_freq
        self.ema_model_alpha = args.stable_model_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.global_step = 0

    def end_task(self, dataset, tb_logger):
        # Step LR decay
        if self.args.slow_lr_multiplier_decay:
            self.slow_lr_multiplier *= self.lr_decay_rate
            print('Slow LR multiplier decayed a bit!')

    def update_ema_model_variables(self):
        alpha_ema = min(1 - 1 / (self.global_step + 1), self.ema_model_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha_ema).add_(1 - alpha_ema, param.data)

    def observe_1(self, inputs, labels, not_aug_inputs, t, lr, cl_mask=None):
        loss_b = torch.tensor(0)
        self.opt.zero_grad()
        outputs = self.net(inputs)

        # loss scores calculated for loss aware balanced reservoir sampling
        if cl_mask is not None:
            mask_add_on = torch.zeros_like(outputs)
            mask_add_on[:, cl_mask] = float('-inf')
            cl_masked_output = outputs + mask_add_on
            loss_scores = self.loss(cl_masked_output, labels, reduction='none')
        else:
            loss_scores = self.loss(outputs, labels, reduction='none')

        loss = loss_scores.mean()
        assert not torch.isnan(loss)
        loss.backward()

        self.update_non_cumulative_sparse()

        if not self.buffer.is_empty():

            self.opt.zero_grad()
            buf_inputs_1, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs_1)
            buf_outputs_ema = self.ema_model(buf_inputs_1)
            loss_b = self.reg_weight * F.mse_loss(buf_outputs, buf_outputs_ema.detach())

            buf_inputs_2, buf_labels_2, buf_indexes_2 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_indexes=True)
            buf_outputs = self.net(buf_inputs_2)
            loss_b_scores = self.loss(buf_outputs, buf_labels_2, reduction='none')
            loss_b += loss_b_scores.mean()

            self.buffer.update_scores(buf_indexes_2, -loss_b_scores.detach())

            assert not torch.isnan(loss_b)
            loss_b.backward()

            # retain gradients for weights in cumulative sparse set
            self.update_cumulative_sparse()

        self.opt.step()
        self.net_grad.zero_grad()

        if self.args.reservoir_buffer:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 loss_scores=-loss_scores.detach())

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()

        return loss.item(), loss_b.item()

    def observe_2(self, inputs, labels, not_aug_inputs, t, lr, cl_mask=None):
        loss_b = torch.tensor(0)
        self.opt.zero_grad()
        # mask out weights not in the 'current' sparse mask
        if self.args.mask_non_sparse_weights:
            self.maskout_non_sparse()

        outputs = self.net(inputs)

        if cl_mask is not None:
            mask_add_on = torch.zeros_like(outputs)
            mask_add_on[:, cl_mask] = float('-inf')
            cl_masked_output = outputs + mask_add_on
            loss_scores = self.loss(cl_masked_output, labels, reduction='none')
        else:
            loss_scores = self.loss(outputs, labels, reduction='none')

        loss = loss_scores.mean()
        assert not torch.isnan(loss)
        loss.backward()

        # Update gradients of current sparse mask that are NOT in sparse set
        self.update_non_overlapping_sparse()

        if not self.buffer.is_empty():
            self.opt.zero_grad()
            # mask out weights not in the 'current' sparse mask
            if self.args.mask_non_sparse_weights:
                self.maskout_non_sparse()

            buf_inputs_1, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs_1)
            buf_outputs_ema = self.ema_model(buf_inputs_1)
            loss_b = self.reg_weight * F.mse_loss(buf_outputs, buf_outputs_ema.detach())

            buf_inputs_2, buf_labels_2, buf_indexes_2 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_indexes=True)
            buf_outputs = self.net(buf_inputs_2)
            loss_b_scores = self.args.beta * self.loss(buf_outputs, buf_labels_2, reduction='none')
            loss_b += loss_b_scores.mean()

            self.buffer.update_scores(buf_indexes_2, -loss_b_scores.detach())

            assert not torch.isnan(loss_b)
            loss_b.backward()

            # Update gradients of current sparse mask that are in sparse set
            self.update_overlapping_sparse()

        self.opt.step()
        self.net.zero_grad()

        if self.args.reservoir_buffer:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 loss_scores=-loss_scores.detach())

        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()

        return loss.item(), loss_b.item()

    def observe_3(self, inputs, labels, not_aug_inputs, t, lr, cl_mask=None):
        loss_b = torch.tensor(0)
        self.opt.zero_grad()
        outputs = self.net(inputs)

        if self.args.mask_cum_sparse_weights:
            self.maskout_cum_sparse()

        if cl_mask is not None:
            mask_add_on = torch.zeros_like(outputs)
            mask_add_on[:, cl_mask] = float('-inf')
            cl_masked_output = outputs + mask_add_on
            loss_scores = self.loss(cl_masked_output, labels, reduction='none')
        else:
            loss_scores = self.loss(outputs, labels, reduction='none')

        loss = loss_scores.mean()
        assert not torch.isnan(loss)
        loss.backward()

        self.update_non_cumulative_sparse(observe=True)

        self.opt.step()
        self.net_grad.zero_grad()

        if self.args.reservoir_buffer:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 loss_scores=-loss_scores.detach())

        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()

        return loss.item(), loss_b.item()
