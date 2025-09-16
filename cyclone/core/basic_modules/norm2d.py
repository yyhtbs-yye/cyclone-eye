import torch
import torch.nn as nn

class ChannelLayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: (B, dim, H, W)
        mean = x.mean(dim=1, keepdim=True)                    # over dim
        var  = x.var(dim=1, keepdim=True, unbiased=False)     # over dim
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight + self.bias
        return y
