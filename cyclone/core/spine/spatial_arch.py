import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import  trunc_normal_

from cyclone.core.basic_modules.patch2d import PatchEmbed2d, PatchUnEmbed2d
from cyclone.core.basic_modules.norm2d import ChannelLayerNorm2d
from cyclone.core.head_n_tail.spatial_vanilla import SpatialProcessor, Upsampler

class BasicResGroup(nn.Module):

    def __init__(self, dim, depth, pre_block_def, main_block_defs, post_block_def):

        super().__init__()
        self.dim = dim
        self.depth = depth
        main_blocks = []

        self.tail_conv = nn.Conv2d(dim, dim, 3, 1, 1)  # HAT’s conv at the end of RHAG

        if pre_block_def is not None and len(pre_block_def) == 2:
            self.pre_block = pre_block_def[0](dim=dim, **pre_block_def[1])
        else:
            self.pre_block = nn.Identity()

        for i in range(depth):
            for main_block_def, main_block_config in main_block_defs:
                main_blocks.append(main_block_def(dim=dim, **main_block_config))

        self.main_blocks = nn.ModuleList(main_blocks)

        if post_block_def is not None and len(post_block_def) == 2:
            self.post_block = post_block_def[0](dim=dim, **post_block_def[1])
        else:
            self.post_block = nn.Identity()

        self.gate = nn.Parameter(torch.full((1, dim, 1, 1), 1.0))

    def forward(self, x, **args):

        shortcut = x
        if not isinstance(self.pre_block, nn.Identity):
            x = self.pre_block(x, **args)

        for main_block in self.main_blocks:
            x = main_block(x, **args)

        if not isinstance(self.post_block, nn.Identity):
            x = self.post_block(x, **args)

        x = self.tail_conv(x)

        return shortcut + x * self.gate

class BasicResEmbeddingGroup(nn.Module):

    def __init__(self, dim, patch_size, depth,
                 pre_block_def, main_block_defs, post_block_def):
        super(BasicResEmbeddingGroup, self).__init__()

        self.dim = dim

        effective_dim = dim * (patch_size ** 2)

        self.residual_group = BasicResGroup(
            dim=effective_dim, depth=depth,
            pre_block_def=pre_block_def,
            main_block_defs=main_block_defs,
            post_block_def=post_block_def
        )

        self.conv = nn.Conv2d(dim, dim, kernel_size=patch_size*2+1, stride=1, padding=patch_size)

        self.patch_embed = PatchEmbed2d(origin_dim=dim, target_dim=None, patch_size=patch_size)

        self.patch_unembed = PatchUnEmbed2d(origin_dim=None, target_dim=dim, patch_size=patch_size)

    def forward(self, x, **args):

        shortcut = x
        x = self.conv(x)                            # to mitigate the patch boundary issues. 
        x = self.patch_embed(x)                     # (B, C*p*p, nH, nW)
        x = self.residual_group(x, **args)    # (B, C*p*p, nH, nW)
        x = self.patch_unembed(x)                   # (B, C, H, W)

        return x + shortcut

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, dim=64, patch_size=1, depths=(6, 6, 6, 6),
                 prep_args=None, pre_block_def=None, main_block_defs=None, post_block_def=None,):

        super(SpatialFeatureExtractor, self).__init__()

        self.dim = dim
        self.patch_size = patch_size

        main_block_defs = [] if main_block_defs is None else list(main_block_defs)

        self.prep_args = prep_args

        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicResEmbeddingGroup(dim=dim,
                                           patch_size=patch_size,
                                           depth=depths[i_layer],
                                           pre_block_def=pre_block_def,
                                           main_block_defs=main_block_defs,
                                           post_block_def=post_block_def,
            )

            self.layers.append(layer)

    def forward(self, x, **args):

        if self.prep_args is not None:
            args = self.prep_args(x, dim=self.dim, patch_size=self.patch_size, **args)

        if not isinstance(args, dict):
            raise TypeError("prep_args must return a dict of kwargs.")

        for layer in self.layers:           
            x = layer(x, **args)      # B C H W

        return x

class VisionRestormer(nn.Module):

    def __init__(self, dim=64, patch_size=1,
                 in_chans=3, out_chans=3,
                 depths=(6, 6, 6, 6),
                 pos_drop=0.0, scale=2, 
                 spatial_proc_def=None, upsample_def=None,
                 pre_block_def=None, main_block_defs=None, post_block_def=None,):
        
        super(VisionRestormer, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.dim = dim
        self.scale = scale
        self.patch_size = patch_size

        self.pos_drop = nn.Dropout2d(p=pos_drop)

        spatial_proc_def = (SpatialProcessor, {}) if spatial_proc_def is None else (spatial_proc_def[0], dict(spatial_proc_def[1]))

        self.spt = spatial_proc_def[0](in_chans=in_chans, dim=dim, **spatial_proc_def[1])

        self.main_block_defs = [] if main_block_defs is None else list(main_block_defs)

        self.tfx = SpatialFeatureExtractor(dim=dim, patch_size=patch_size, depths=depths, 
                                           pre_block_def=pre_block_def, 
                                           main_block_defs=main_block_defs, 
                                           post_block_def=post_block_def)

        upsample_def = (Upsampler, {}) if upsample_def is None else (upsample_def[0], dict(upsample_def[1]))

        self.up = upsample_def[0](scale=scale, dim=dim, out_chans=out_chans, **upsample_def[1])

        self.norm = ChannelLayerNorm2d(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, ChannelLayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, **args):

        x = self.spt(x)            # B C H W
        shortcut = x

        x = self.pos_drop(x)                # B C H W

        x = self.tfx(x, **args)

        x = self.norm(x)                    # B C H W

        x = self.up(x + shortcut)                # B C H W

        return x
