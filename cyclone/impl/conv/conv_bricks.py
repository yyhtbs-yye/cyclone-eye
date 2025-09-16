import torch
import torch.nn as nn
from timm.layers import ConvNormAct

from cyclone.core.spine.spatial_arch import VisionRestormer

class ConvPreBlock(nn.Module):
    """
    Optional pre-block: two 3x3 Conv-Norm-Act layers at the start of a group.
    Signature matches your framework: __init__(dim, **cfg), forward(x, **args)
    """
    def __init__(self, dim: int, act_layer='relu', norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(dim, dim, kernel_size=3, padding=1,
                        act_layer=act_layer, norm_layer=norm_layer, bias=True),
            ConvNormAct(dim, dim, kernel_size=3, padding=1,
                        act_layer=act_layer, norm_layer=norm_layer, bias=True),
        )

    def forward(self, x, **_):
        return self.net(x)

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        dim (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, dim=64, bias=True, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        self.acti = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.conv2(self.acti(self.conv1(x)))
        return identity + out * self.res_scale


class ConvPostBlock(nn.Module):
    """
    Optional post-block: one 3x3 Conv-Norm-Act to polish features after the main stack.
    """
    def __init__(self, dim: int, act_layer='relu', norm_layer='bn2d'):
        super().__init__()
        self.net = ConvNormAct(dim, dim, kernel_size=3, padding=1,
                               act_layer=act_layer, norm_layer=norm_layer, bias=True)

    def forward(self, x, **_):
        return self.net(x)
