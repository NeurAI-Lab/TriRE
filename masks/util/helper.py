import torch
import torch.nn as nn


def renint_usnig_method(param_data, mask, method='xavier'):
    if method == 'kaiming':
        nn.init.kaiming_uniform_(mask)
    elif method == 'normal':
        std, mean = torch.std_mean(param_data)
        nn.init.normal_(mask, mean, std)
    elif method == 'xavier':
        nn.init.xavier_uniform_(mask)


def re_init_weights(param_data, shape, device, reinint_method='xavier'):
    mask = torch.empty(shape, requires_grad=False, device=device)
    if len(mask.shape) < 2:
        mask = torch.unsqueeze(mask, 1)
        renint_usnig_method(param_data, mask, reinint_method)
        mask = torch.squeeze(mask, 1)
    else:
        renint_usnig_method(param_data, mask, reinint_method)
    return mask