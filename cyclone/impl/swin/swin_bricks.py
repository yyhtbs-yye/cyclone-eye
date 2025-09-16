import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import ConvNormAct
from timm.layers import to_2tuple, trunc_normal_
from einops import rearrange

from cyclone.impl.swin.swin_utils import compute_attn_mask, window_partition_nchw, window_reverse_nchw
from cyclone.core.basic_modules.norm2d import ChannelLayerNorm2d
from cyclone.core.basic_modules.msa2d import MaskAttention2d
from cyclone.core.basic_utils.rpi import calculate_rpi_sa

class PostConv(nn.Module):
    def __init__(self, dim, mode='1conv'):
        super().__init__()
        if mode == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif mode == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), 
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x):
        return self.conv(x)

class SwinAttention(nn.Module):
    """
    Swin Attention that accepts:
      - dim
      - window_size (tuple[int, int] or int)
      - shift_size (tuple[int, int] or int)
      - num_heads
      - qkv_bias, qk_scale, attn_drop, proj_drop
    Forward expects x of shape (B, C, H, W). Returns (B, C, H, W).
    """
    def __init__(self, dim, window_size=7, shift_size=0, num_heads=4,
                 qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.shift_size = to_2tuple(shift_size)
        self.num_heads = num_heads

        # sanity for shift
        wH, wW = self.window_size
        sH, sW = self.shift_size

        assert sH < wH and sW < wW

        relative_position_index = calculate_rpi_sa(self.window_size).view(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn = MaskAttention2d(dim, num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        wH, wW = self.window_size
        sH, sW = self.shift_size

        # cyclic shift
        if sH > 0 or sW > 0:
            shifted_x = torch.roll(x, shifts=(-sH, -sW), dims=(2, 3))
        else:
            shifted_x = x

        # windows
        x_windows = window_partition_nchw(shifted_x, (wH, wW))  # (B*nW, C, wH, wW)

        # attn mask
        attn_mask = compute_attn_mask(H, W, self.window_size, self.shift_size)

        if attn_mask is not None:
            attn_mask = attn_mask.to(x_windows.device)
            if attn_mask.numel() == 0:  # optional: drop the sentinel path entirely
                attn_mask = None

        # relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index]
        rpe_bias = rearrange(bias, '(W O) C -> C W O', W=self.window_size[0]*self.window_size[1], 
                                                         O=self.window_size[0]*self.window_size[1]).unsqueeze(0) 

        # attention
        attn_windows = self.attn(x_windows, mask=attn_mask, rpe_bias=rpe_bias)  # (B*nW, C, wH, wW)

        # merge windows
        shifted_x = window_reverse_nchw(attn_windows, (wH, wW), B, H, W)  # (B, C, H, W)

        # reverse shift
        if sH > 0 or sW > 0:
            x = torch.roll(shifted_x, shifts=(sH, sW), dims=(2, 3))
        else:
            x = shifted_x

        return x

class STLBlock(nn.Module):
    """
    Main Swin Transformer Layer block: (internal LN in MSA) -> MSA + x -> LN -> MLP + x
    """
    def __init__(self, dim, num_heads, window_size, shift_size,
                 qkv_bias=True, qk_scale=None,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, norm_layer=ChannelLayerNorm2d):
        super().__init__()

        self.ln1 = norm_layer(dim)

        # SwinTransformerBlock signature: (dim, window_size, shift_size, num_heads, ...)
        self.msa = SwinAttention(
            dim=dim, window_size=window_size, shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
            num_heads=num_heads, drop=drop, attn_drop=attn_drop)
        
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
