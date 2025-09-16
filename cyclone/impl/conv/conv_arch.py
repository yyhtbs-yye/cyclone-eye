import torch
import torch.nn as nn
from timm.layers import ConvNormAct

from cyclone.core.spine.spatial_arch import VisionRestormer
from cyclone.impl.conv.conv_bricks import ConvPreBlock, ResidualBlockNoBN, ConvPostBlock

def build_model(**model_config):
    """
    Returns an instance of YOUR VisionRestormer, configured to use the pure Conv components above.
    - Uses patch_size=1 (no patch embedding effect).
    - args are ignored by all blocks; pass nothing at call time.
    - You can vary depths/scale/dim as usual.
    """

    pre_block_def   = (ConvPreBlock,   dict(act_layer=model_config['act_layer'],  norm_layer=model_config['norm_layer']))
    main_block_defs = [(ResidualBlockNoBN, dict())]
    post_block_def  = (ConvPostBlock,  dict(act_layer=model_config['act_layer'],  norm_layer=model_config['norm_layer']))

    # patch_size=1 ensures the embedding/unembedding path is effectively identity in your framework
    model = VisionRestormer(
        dim=model_config['dim'],
        patch_size=model_config['patch_size'],
        in_chans=model_config['in_chans'],
        out_chans=model_config['out_chans'],
        depths=model_config['depths'],
        pos_drop=model_config['pos_drop'],
        scale=model_config['scale'],
        pre_block_def=pre_block_def,
        main_block_defs=main_block_defs,
        post_block_def=post_block_def,
    )
    return model

model = build_model(
    dim=64,
    patch_size=1,
    in_chans=3,
    out_chans=3,
    depths=(3, 3, 3, 3),
    scale=2,
    pos_drop=0.0,
    act_layer='relu',    # or 'gelu', etc. (timm strings work)
    norm_layer='batchnorm',   # or 'in2d', 'gn', or None (ConvNormAct adapts)
)

if __name__=="__main__":
    import torch
    import torch.nn as nn
    from torchviz import make_dot
    from thop import profile, clever_format

    # --- Replace with your real input shape ---
    dummy = torch.randn(1, 3, 128, 128)

    # 1) Visualize the computation graph
    y = model(dummy)
    dot = make_dot(y, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True)
    dot.format = "svg"
    dot.render("model_graph", cleanup=True)  # creates model_graph.svg

    # 2) FLOPs/MACs + params
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    # Note: FLOPs are often reported as 2×MACs for conv/GEMM
    flops = "≈ " + str(clever_format([2*eval(macs.replace('G','e9').replace('M','e6').replace('K','e3')), 0], "%.3f")[0]) + " FLOPs"  # quick estimate

    print(f"Params: {params}")
    print(f"MACs:   {macs}")
    print(f"FLOPs:  {flops}")
    print("Saved graph to model_graph.svg")