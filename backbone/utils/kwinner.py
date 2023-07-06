import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SparsifyBase(nn.Module):
    def __init__(self, sparse_ratio=0.5):
        super(SparsifyBase, self).__init__()
        self.sr = sparse_ratio
        self.preact = None
        self.act = None

    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()

        return hook

    def record_activation(self):
        self.register_forward_hook(self.get_activation())


class Sparsify1D(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify1D, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        k = int(self.sr * x.shape[1])
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify1D_kactive(SparsifyBase):
    def __init__(self, k=1):
        super(Sparsify1D_kactive, self).__init__()
        self.k = k

    def forward(self, x):
        k = self.k
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D(SparsifyBase):
    def __init__(self, sparse_ratio=0.3):
        super(Sparsify2D, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        tmpx = x.view(x.shape[0], x.shape[1], -1)
        topval = tmpx.topk(k, dim=2)[0][:, :, -1]
        topval = topval.expand(x.shape[2], x.shape[3], x.shape[0], x.shape[1]).permute(2, 3, 0, 1)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D_vol(SparsifyBase):
    """cross channel sparsify"""

    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_vol, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        size = x.shape[1] * x.shape[2] * x.shape[3]
        k = int(self.sr * size)

        tmpx = x.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).view_as(x)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D_all(SparsifyBase):
    """cross channel sparsify"""

    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_all, self).__init__()
        self.sr = sparse_ratio
        self.act_count = None

    def update_count(self, mask):
        if self.act_count is None:
            self.act_count = mask
        else:
            self.act_count += mask

    def forward(self, x):
        size = x.shape[1]
        k = int(self.sr * size)
        tmpx = x.view(x.shape[0], x.shape[1], -1)
        topvalues = tmpx.topk(1, dim=2)[0].squeeze(2)
        topval, indices = topvalues.topk(k, dim=1)
        act_count = torch.zeros(topvalues.shape, device=topvalues.device)
        self.update_count(act_count.scatter_(1, indices, 1).sum(dim=0))
        comp = act_count.unsqueeze(2).repeat(1, 1, tmpx.shape[2]).view_as(x)
        return comp * x


class Sparsify2D_kactive(SparsifyBase):
    """cross channel sparsify"""

    def __init__(self, k=4):
        super(Sparsify2D_kactive, self).__init__()
        self.k = k

    def forward(self, x):
        k = self.k
        tmpx = x.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).view_as(x)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D_abs(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_abs, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        absx = torch.abs(x)
        tmpx = absx.view(absx.shape[0], absx.shape[1], -1)
        topval = tmpx.topk(k, dim=2)[0][:, :, -1]
        topval = topval.expand(absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]).permute(2, 3, 0, 1)
        comp = (absx >= topval).to(x)
        return comp * x


class Sparsify2D_invabs(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_invabs, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        absx = torch.abs(x)
        tmpx = absx.view(absx.shape[0], absx.shape[1], -1)
        topval = tmpx.topk(k, dim=2, largest=False)[0][:, :, -1]
        topval = topval.expand(absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]).permute(2, 3, 0, 1)
        comp = (absx >= topval).to(x)
        return comp * x

