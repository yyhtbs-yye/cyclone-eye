import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import to_2tuple

class MaskAttention2dOp(nn.Module):
    """
    Masked multi-head self attention (W-MSA) Operation (without Matrix QKV) with relative position bias for window attention.

    Input:
      q, k, v: (B*nW, C, H, W)  where H, W are the window spatial sizes (wH, wW)
      mask: (nW, H*W, H*W) or None
      rpe_bias: (1, nH, H*W, H*W) or None
    Output:
      (B*nW, C, H, W)
    """
    def __init__(self, dim, num_heads, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or (head_dim ** -0.5)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, mask=None, rpe_bias=None):
        """
        x:    (B*nW, C, H, W) with H, W = window_size
        mask: (nW, H*W, H*W) or None
        """
        Bn, C, H, W = q.shape
        N = H * W

        # scale q
        q = q * self.scale

        # flatten spatial only (no permute)
        # shapes: (Bn, heads, head_dim, (H W) -> N)
        q = q.view(Bn, self.num_heads, C // self.num_heads, N)
        k = k.view(Bn, self.num_heads, C // self.num_heads, -1)
        v = v.view(Bn, self.num_heads, C // self.num_heads, -1)

        # attention logits: (Bn, heads, N, N) using einsum to avoid explicit transpose/permute
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k)  # sum over head_dim

        if rpe_bias is not None:

            attn = attn + rpe_bias

        # mask: (nW, N, N) -> broadcast over batch groups
        if mask is not None:
            nW = mask.shape[0]
            assert (Bn % nW) == 0, "Bn must be divisible by nW"
            B = Bn // nW
            attn = attn.view(B, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)  # (B, nW, heads, N, N)
            attn = attn.view(Bn, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # weighted sum: (Bn, heads, head_dim, N)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)

        # fold heads back to channels and restore spatial H, W
        out = out.reshape(Bn, C, H, W)  # (Bn, C, H, W)

        # final projection
        out = self.proj_conv(out)
        out = self.proj_drop(out)
        return out


class MaskAttention2d(nn.Module):
    """
    Window based multi-head self attention (W-MSA) with relative position bias.

    Input:
      x: (B*nW, C, H, W)  where H, W are the window spatial sizes (wH, wW)
      mask: (nW, H*W, H*W) or None
    Output:
      (B*nW, C, H, W)
    """
    def __init__(self, dim, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or (head_dim ** -0.5)
        
        self.msaop = MaskAttention2dOp(dim=dim, num_heads=num_heads, qk_scale=qk_scale, 
                                     attn_drop=attn_drop, proj_drop=proj_drop)

        # 1x1 conv for qkv and projection (channel-first friendly)
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)

    def forward(self, x, mask=None, rpe_bias=None):
        """
        x:    (B*nW, C, H, W) with H, W = window_size
        mask: (nW, H*W, H*W) or None
        """
        Bn, C, H, W = x.shape

        qkv = self.qkv_conv(x)  # (Bn, 3*C, H, W)
        q, k, v = torch.split(qkv, [C, C, C], dim=1)

        out = self.msaop(q, k, v, mask, rpe_bias)

        return out