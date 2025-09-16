from functools import lru_cache
import torch
from timm.layers import ndgrid, to_2tuple
from einops import rearrange

@lru_cache(maxsize=10)
def compute_attn_mask(Hp, Wp, window_size, shift_size):

    wH, wW = window_size
    sH, sW = shift_size

    if sH == 0 and sW == 0:
        return None
    else:
        img_mask = torch.zeros((1, Hp, Wp, 1))  # (1, Hp, Wp, 1)
        cnt = 0
        h_slices = (slice(0, -wH), slice(-wH, -sH) if sH > 0 else slice(-wH, None), slice(-sH, None) if sH > 0 else slice(-0, None))
        w_slices = (slice(0, -wW), slice(-wW, -sW) if sW > 0 else slice(-wW, None), slice(-sW, None) if sW > 0 else slice(-0, None))

        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = rearrange(
            img_mask,
            'b (nh h) (nw w) c -> (b nh nw) (h w) c',
            h=wH, w=wW
        )
        mask_windows = mask_windows.view(-1, wH * wW)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        mask = attn_mask  # (nW, N, N)

        return mask

# ----------------------------------utils-------------------------------------------------
def window_partition_nchw(x, window_size):  # x: (B, C, H, W)
    B, C, H, W = x.shape
    wH, wW = to_2tuple(window_size)
    assert H % wH == 0 and W % wW == 0
    nH, nW = H // wH, W // wW
    # reshape then reorder window blocks next to batch dimension
    x = x.view(B, C, nH, wH, nW, wW)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()     # (B, nH, nW, C, wH, wW)
    x = x.view(B * nH * nW, C, wH, wW)               # (B*nW, C, wH, wW)
    return x

def window_reverse_nchw(windows, window_size, B, H, W):
    # windows: (B*nW, C, wH, wW)
    C = windows.shape[1]
    wH, wW = to_2tuple(window_size)
    nH, nW = H // wH, W // wW
    x = windows.view(B, nH, nW, C, wH, wW)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()     # (B, C, nH, wH, nW, wW)
    x = x.view(B, C, H, W)
    return x