import torch
import torch.nn as nn

from cyclone.core.spine.spatial_arch import VisionRestormer
from cyclone.core.basic_modules.norm2d import ChannelLayerNorm2d
from cyclone.impl.hat.hat_bricks import OCABlock, HABlock

class HAT(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, 
                 dim=64, num_heads=16, patch_size=1, 
                 depths=(3, 3, 3, 3), scale=2, pos_drop=0.0,
                 window_size=8, shift_size=4, overlap_ratio=0.5,
                 compress_ratio=4, squeeze_factor=16,
                 qkv_bias=True, qk_scale=None,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, norm_layer=ChannelLayerNorm2d):
        super().__init__()

        pre_block_def = None

        main_block_defs = [(HABlock, dict(num_heads=num_heads, window_size=window_size, shift_size=shift_size, 
                                          mlp_ratio=mlp_ratio,
                                          compress_ratio=compress_ratio, squeeze_factor=squeeze_factor, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                          norm_layer=norm_layer)),
                           (HABlock, dict(num_heads=num_heads, window_size=window_size, shift_size=0, 
                                          mlp_ratio=mlp_ratio,
                                          compress_ratio=compress_ratio, squeeze_factor=squeeze_factor, 
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                          norm_layer=norm_layer)),]

        post_block_def = (OCABlock, dict(num_heads=num_heads, window_size=window_size, overlap_ratio=overlap_ratio,
                                         mlp_ratio=mlp_ratio, norm_layer=norm_layer))

        self.model = VisionRestormer(
            dim=dim,
            patch_size=patch_size,
            in_chans=in_chans,
            out_chans=out_chans,
            depths=depths,
            pos_drop=pos_drop,
            scale=scale,
            pre_block_def=pre_block_def,
            main_block_defs=main_block_defs,
            post_block_def=post_block_def,
        )

    def forward(self, x):

        return self.model(x)

if __name__=="__main__":
    import torch
    import torch.nn as nn
    from thop import profile, clever_format

    # --- Replace with your real input shape ---
    dummy = torch.randn(1, 3, 128, 128)

    model = HAT()

    # 2) FLOPs/MACs + params
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    # Note: FLOPs are often reported as 2×MACs for conv/GEMM
    flops = "≈ " + str(clever_format([2*eval(macs.replace('G','e9').replace('M','e6').replace('K','e3')), 0], "%.3f")[0]) + " FLOPs"  # quick estimate

    print(f"Params: {params}")
    print(f"MACs:   {macs}")
    print(f"FLOPs:  {flops}")