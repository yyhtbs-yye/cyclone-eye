import torch.nn as nn
from timm.layers import to_2tuple

from cyclone.core.basic_modules.norm2d import ChannelLayerNorm2d
from cyclone.core.basic_modules.msa2d import MaskAttention2d
from cyclone.core.basic_modules.patch2d import PatchEmbed2d, PatchUnEmbed2d

from cyclone.core.basic_modules.drpi2d import DynamicPosEncoder

class PatchAttention(nn.Module):
    """
    Main VIT block: (internal LN in MSA) -> MSA + x -> LN -> MLP + x
    """
    def __init__(self, dim, num_heads, patch_size,
                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0):
        super().__init__()

        self.patch_size = to_2tuple(patch_size)

        pp = self.patch_size[0] * self.patch_size[1]

        self.vdim = dim * pp

        self.pos = DynamicPosEncoder(dim, num_heads)

        self.attn = MaskAttention2d(dim, num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop)
        
        self.patch_embed = PatchEmbed2d(patch_size, origin_dim=dim, target_dim=dim)

        self.patch_unembed = PatchUnEmbed2d(patch_size, origin_dim=dim, target_dim=dim)

    def forward(self, x):

        x = self.patch_embed(x)

        _, C, H, W = x.shape

        position_bias = self.pos(H, W, x.device)

        x = self.attn(x, rpe_bias=position_bias)

        x = self.patch_unembed(x)

        return x

class VITBlock(nn.Module):
    """
    Main VIT block: (internal LN in MSA) -> MSA + x -> LN -> MLP + x
    """
    def __init__(self, dim, num_heads, patch_size, 
                 qkv_bias=True, qk_scale=None,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, norm_layer=ChannelLayerNorm2d):
        super().__init__()

        self.patch_size = to_2tuple(patch_size)

        self.ln1 = norm_layer(dim)

        self.pa = PatchAttention(dim, num_heads, patch_size=patch_size,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 attn_drop=attn_drop, drop=drop)
        
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
        x = self.pa(self.ln1(x)) + x

        # Second residual block: LN -> MLP
        x = self.mlp(self.ln2(x)) + x
        return x
