import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbed2d(nn.Module):
    """
    Space-to-depth (+ optional 1x1 conv projection).

    In:  x.shape (B, C, H, W)  with H,W divisible by p
    Do:  rearrange -> (B, C*p*p, H/p, W/p), then optional 1x1 Conv2d to target_dim
    Out: tokens ∈ (B, C_embed, H/p, W/p)
    """
    def __init__(self, patch_size=4, origin_dim=64, target_dim=None, bias=True):
        super().__init__()
        self.patch_size = int(patch_size)

        in_channels = origin_dim * self.patch_size * self.patch_size
        if target_dim is None:
            self.proj = nn.Identity()  # no channel projection
        else:
            self.proj = nn.Conv2d(in_channels, target_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "H and W must be divisible by patch_size"

        # (B, C, H, W) -> (B, C*p*p, H/p, W/p)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=p, p2=p)

        # Optional channel projection with 1x1 conv
        x = self.proj(x)  # (B, target_dim or C*p*p, H/p, W/p)
        return x

class PatchUnEmbed2d(nn.Module):
    """
    Depth-to-space with optional 1x1 conv pre-expansion.

    In:  tokens.shape (B, C_embed, H/p, W/p)
    Do:  optional 1x1 Conv2d: C_embed -> (target_dim*p*p), then rearrange -> (B, target_dim, H, W)
    Out: x.shape (B, target_dim, H, W)

    If origin_dim is None, no pre-expansion is done (assumes tokens already have C = target_dim*p*p).
    """
    def __init__(self, patch_size=4, origin_dim=None, target_dim=64, bias=True):
        super().__init__()
        self.patch_size = int(patch_size)

        if origin_dim is None:
            # No projection: expect input channels == target_dim * p*p
            self.proj = nn.Identity()
        else:
            out_channels = target_dim * self.patch_size * self.patch_size
            self.proj = nn.Conv2d(origin_dim, out_channels, kernel_size=1, bias=bias)

        self.target_dim = target_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, C_in, H_p, W_p)
        p = self.patch_size

        # Optional channel projection to target_dim * p^2
        x = self.proj(tokens)  # (B, target_dim*p*p, H_p, W_p)

        # Robustness: enforce channel contract
        C = x.shape[1]
        expected = self.target_dim * p * p
        assert C == expected, f"channels={C}, expected {expected} (=target_dim·p²)"

        # (B, target_dim*p*p, H_p, W_p) -> (B, target_dim, H_p*p, W_p*p)
        x = rearrange(x, 'b (c p1 p2) h w -> b c (h p1) (w p2)', p1=p, p2=p)
        return x
