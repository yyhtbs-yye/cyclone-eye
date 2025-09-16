import torch
import torch.nn as nn
from timm.layers import to_2tuple
from einops import rearrange

from .hma_utils import grid_shuffle, grid_unshuffle
from cyclone.core.basic_modules.msa2d import MaskAttention2dOp
from cyclone.core.basic_modules.attention2d import ChannelAttention2d
from cyclone.core.basic_modules.norm2d import ChannelLayerNorm2d
from cyclone.core.basic_modules.drpi2d import DynamicPosEncoder
from cyclone.impl.swin.swin_bricks import SwinAttention, STLBlock

class GridAttention(nn.Module):
    # """ Grid Attention Module """

    def __init__(self, dim, num_heads, interval_size, 
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.interval_size = interval_size

        self.dim = dim

        self.qkv_conv = nn.Conv2d(dim, self.dim * 3, kernel_size=1, bias=qkv_bias)

        self.grid_proj = nn.Conv2d(dim, self.dim, kernel_size=1)

        self.pos = DynamicPosEncoder(dim, num_heads)

        self.attn1_op = MaskAttention2dOp(dim, num_heads=num_heads, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.attn2_op = MaskAttention2dOp(dim, num_heads=num_heads, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x):

        B, C, H, W = x.shape

        qkv = self.qkv_conv(x)                                          # B, C*3, H, W

        x_grid = self.grid_proj(grid_shuffle(x, self.interval_size))    # B*K*K, C, H//K, W//K

        qkv = grid_shuffle(qkv, self.interval_size)                     # B*K*K, C*3, H//K, W//K
        
        position_bias = self.pos(H // self.interval_size, W // self.interval_size, x_grid.device)

        q, k, v = torch.split(qkv, [self.dim, self.dim, self.dim], dim=1)  # each: B*K*K, C, H//K, W//K

        x = self.attn1_op(x_grid, k, v, rpe_bias=position_bias)                     # B*K*K, C, H//K, W//K
        x = self.attn2_op(q, x_grid, x, rpe_bias=position_bias)                     # B*K*K, C, H//K, W//K
                
        x_grid_attn = grid_unshuffle(x, self.interval_size)               # B, C, H, W

        return x_grid_attn

class GABlock(nn.Module):
    # """ Grid Attention Block. """

    def __init__(self, dim, num_heads, interval_size, window_size, shift_size=None, 
                 qkv_bias=True, qk_scale=None, attn_drop=0., drop=0., mlp_ratio=2, norm_layer=ChannelLayerNorm2d):
        super().__init__()
        self.window_size = to_2tuple(window_size)

        self.shift_size = (self.window_size[0] // 2, self.window_size[1] // 2) if shift_size is None else to_2tuple(shift_size)

        self.interval_size = interval_size
        self.ln = norm_layer(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)

        self.grid_attn = GridAttention(dim // 2, num_heads=num_heads // 2, 
                                       interval_size=interval_size, qk_scale=qk_scale, attn_drop=attn_drop)
        
        self.swin_attn = SwinAttention(dim // 4, window_size=self.window_size, shift_size=self.shift_size,
                                       num_heads=num_heads // 2, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop=drop)
        
        self.win_attn = SwinAttention(dim // 4, window_size=self.window_size, shift_size=0,
                                      num_heads=num_heads // 2, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop=drop)
        
        self.fc = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Dropout(drop),
        )

    def forward(self, x):
        
        B, C, H, W = x.shape

        shortcut = x

        x_grid, x_swin, x_win = torch.split(x, [C // 2, C // 4, C // 4], dim=1)

        x_grid_attn = self.grid_attn(x_grid)

        x_swin_attn = self.swin_attn(x_swin)

        x_win_attn = self.win_attn(x_win)

        x = torch.cat([x_win_attn, x_swin_attn, x_grid_attn], dim=1)

        x = self.ln(self.fc(x))

        x = shortcut + x

        x = x + self.norm2(self.mlp(x)) + shortcut

        return x

class FusedConv(nn.Module):

    def __init__(self, num_feat, expand_size=4, attn_ratio=4, depthwise=False, conv_bias=True, norm_layer=ChannelLayerNorm2d):
        super(FusedConv, self).__init__()
        mid_feat = num_feat * expand_size
        rd_feat = max(1, mid_feat // attn_ratio)
        self.pre_norm = norm_layer(num_feat)
        self.fused_conv = nn.Conv2d(
            num_feat, mid_feat, kernel_size=3, stride=1, padding=1,
            groups=(num_feat if depthwise else 1), bias=conv_bias
        )
        self.ln = norm_layer(mid_feat)
        self.act1 = nn.GELU()
        self.ca = ChannelAttention2d(mid_feat, rd_feat, bias=True)
        self.mlp  = nn.Conv2d(mid_feat, num_feat, kernel_size=1, bias=conv_bias)

    def forward(self, x):
        shortcut = x
        x = self.pre_norm(x)
        x = self.fused_conv(x)
        x = self.act1(self.ln(x))
        x = self.ca(x)
        x = self.mlp(x)

        return x + shortcut
    
class FABlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=None, expand_size=4, attn_ratio=4,
                 qkv_bias=True, qk_scale=None, attn_drop=0., drop=0., mlp_ratio=2, norm_layer=ChannelLayerNorm2d):
        super(FABlock, self).__init__()

        self.window_size = to_2tuple(window_size)

        self.shift_size = (self.window_size[0] // 2, self.window_size[1] // 2) if shift_size is None else to_2tuple(shift_size)

        self.fused_conv = FusedConv(num_feat=dim, expand_size=expand_size, attn_ratio=attn_ratio, norm_layer=norm_layer)

        self.swin_stl = STLBlock(dim, num_heads, window_size, self.shift_size, 
                                 qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop=drop, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        self.win_stl = STLBlock(dim, num_heads, window_size, 0, 
                                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop=drop, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

    def forward(self, x):

        x = self.fused_conv(x)

        x = self.win_stl(x)

        x = self.swin_stl(x)

        return x
