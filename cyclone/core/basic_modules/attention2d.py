import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention2d(nn.Module):
    """
    Channel Attention (CBAM-style):
    Uses both average- and max-pooled channel descriptors passed through
    a shared MLP (1x1 convs), then gates with sigmoid.
    """
    def __init__(self, dim, rd_dim=None, bias=True, use_maxpool=False):
        super().__init__()
        rd = rd_dim if rd_dim is not None else max(1, dim // 16)
        self.use_maxpool = use_maxpool
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, rd, kernel_size=1, bias=bias),
            nn.SiLU(inplace=True),
            nn.Conv2d(rd, dim, kernel_size=1, bias=bias),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        if self.use_maxpool:
            max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
            attn = self.gate(avg_out + max_out)
        else:
            attn = self.gate(avg_out)
        return x * attn
