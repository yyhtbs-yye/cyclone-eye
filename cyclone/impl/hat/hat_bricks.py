import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.layers import to_2tuple

from cyclone.impl.swin.swin_bricks import SwinAttention, window_partition_nchw, window_reverse_nchw
from cyclone.impl.hat.hat_utils import calculate_rpi_oca

from cyclone.core.basic_modules.norm2d import ChannelLayerNorm2d
from cyclone.core.basic_modules.attention2d import ChannelAttention2d
from cyclone.core.basic_modules.msa2d import MaskAttention2dOp

class CAB(nn.Module):

    def __init__(self, dim, compress_ratio=4, squeeze_factor=16):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(dim, dim // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim // compress_ratio, dim, 3, 1, 1),
            ChannelAttention2d(dim, squeeze_factor))

    def forward(self, x):
        return self.cab(x)
    
class OverlapCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, overlap_ratio,
                qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super(OverlapCrossAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.window_size = to_2tuple(window_size)
        self.overlap_ratio = overlap_ratio
        self.overlap_win_size = (int(self.window_size[0] * overlap_ratio) + self.window_size[0],
                                 int(self.window_size[1] * overlap_ratio) + self.window_size[1])
        self.gap_padding_size = (int(self.window_size[0] * overlap_ratio)//2,
                                 int(self.window_size[1] * overlap_ratio)//2)

        self.attn_op = MaskAttention2dOp(dim, num_heads, qk_scale, attn_drop, proj_drop)

        self.unfold = nn.Unfold(kernel_size=self.overlap_win_size, 
                                stride=self.window_size, 
                                padding=self.gap_padding_size)

        relative_position_index = calculate_rpi_oca(self.window_size, overlap_ratio).view(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((self.window_size[0] + self.overlap_win_size[0] - 1) * (self.window_size[1] + self.overlap_win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.qkv_conv(x)                                      # B, 3*C, H, W

        q, kv = torch.split(qkv, [self.dim, self.dim*2], dim=1)     # B, [C, 2*C], H, W

        q_windows = window_partition_nchw(q, self.window_size)      # B*nW, C, wH, wW

        kv_windows = rearrange(self.unfold(kv),
                               'B (Z C oH oW) nW -> Z (B nW) C (oH oW)', 
                               Z=2, C=C, oH=self.overlap_win_size[0], oW=self.overlap_win_size[1]) # B, C*oH*oW, nW
        
        k_windows = kv_windows[0]
        v_windows = kv_windows[1]

        # relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index]
        rpe_bias = rearrange(bias, '(W O) C -> C W O', W=self.window_size[0]*self.window_size[1], 
                                                         O=self.overlap_win_size[0]*self.overlap_win_size[1]).unsqueeze(0) 

        o = self.attn_op(q_windows, k_windows, v_windows, rpe_bias=rpe_bias)

        y = window_reverse_nchw(o, self.window_size, B, H, W)  # (B, C, H, W)

        return y

class OCABlock(nn.Module):
    """
    Main Overlap Cross Attention block: (internal LN in MSA) -> OCA + x -> LN -> MLP + x
    """
    def __init__(self, dim, num_heads, window_size, overlap_ratio,
                 qkv_bias=True, qk_scale=None,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, norm_layer=ChannelLayerNorm2d):
        super().__init__()

        self.ln1 = norm_layer(dim)

        # SwinTransformerBlock signature: (dim, window_size, shift_size, num_heads, ...)
        self.msa = OverlapCrossAttention(dim, num_heads, window_size, overlap_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop)
        
        self.ln2 = norm_layer(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # First residual block: LN -> MSA
        x = self.msa(self.ln1(x)) + x

        # Second residual block: LN -> MLP
        x = self.mlp(self.ln2(x)) + x
        return x

class HABlock(nn.Module):
    """
    Main Hybrid Attention Block: (internal LN in MSA) -> MSA + x -> LN -> MLP + x
    """
    def __init__(self, dim, num_heads, window_size, shift_size,
                 compress_ratio=4, squeeze_factor=16, gamma_cab=0.01,
                 qkv_bias=True, qk_scale=None,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, norm_layer=ChannelLayerNorm2d):
        super().__init__()

        self.ln1 = norm_layer(dim)

        self.msa = SwinAttention(
            dim=dim, window_size=window_size, shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
            num_heads=num_heads, drop=drop, attn_drop=attn_drop)
        
        self.cab = CAB(dim=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.gamma_cab = gamma_cab
        
        self.ln2 = norm_layer(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # First residual block: LN -> MSA+γCAB
        u = self.ln1(x)
        x = self.msa(u) + self.gamma_cab * self.cab(u) + x

        # Second residual block: LN -> MLP
        x = self.mlp(self.ln2(x)) + x
        return x
