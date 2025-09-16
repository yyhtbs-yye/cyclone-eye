import torch
import torch.nn as nn
from einops import rearrange
from cyclone.core.basic_utils.rpi import calculate_rpi_sa
class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class DynamicPosEncoder(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.idx_cache = {}

    def forward(self, H, W, device):

        nT = H * W

        if (H, W) in self.idx_cache:
            relative_position_index, biases = self.idx_cache[(H, W)]

        else:
            relative_position_index = calculate_rpi_sa((H, W)).to(device)
            biases = torch.stack(torch.meshgrid(
                torch.arange(1 - H, H, device=device),
                torch.arange(1 - W, W, device=device),
                indexing='ij'
            ), dim=-1).reshape(-1, 2).float()
            self.idx_cache[(H, W)] = (relative_position_index, biases)

        rpi_table = self.pos(biases)  # 2H-1 * 2W-1, heads

        relative_position_bias = rpi_table[relative_position_index.view(-1)].view(nT, nT, -1)  # H*W, H*W, nH
        relative_position_bias = rearrange(relative_position_bias, 'nT1 nT2 nH -> 1 nH nT1 nT2')

        return relative_position_bias