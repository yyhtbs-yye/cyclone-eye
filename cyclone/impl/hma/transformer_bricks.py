import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed2d(nn.Module):
    """
    (B, C, H, W) --Unfold--> (B, L, C*p*p) --Linear--> (B, L, embed_dim)
    Returns tokens and (H, W) so you can fold back later.
    """
    def __init__(self, patch_size=4, origin_dim=64, target_dim=None, bias=True):
        super().__init__()
        self.patch_size = patch_size
        if target_dim is None:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(origin_dim * patch_size * patch_size, target_dim, bias=bias)

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W), H and W divisible by patch_size
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size, padding=0)          # (B, C*p*p, L)
        patches = patches.transpose(1, 2) # (B, L, C*p*p)
        tokens = self.proj(patches)       # (B, L, embed_dim)
        nH = H // self.patch_size
        nW = W // self.patch_size
        tokens = tokens.view(B, nH, nW, -1)  # (B, nH, nW, embed_dim)
        return tokens                        # (B, nH, nW, embed_dim)

class PatchUnEmbed2d(nn.Module):
    """
    (B, L, embed_dim) --Linear--> (B, C*p*p, L) --Fold--> (B, C, H, W)
    """
    def __init__(self, patch_size=4, origin_dim=None, target_dim=64, bias=True):
        super().__init__()
        self.patch_size = patch_size
        if origin_dim is None:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(origin_dim, target_dim * patch_size * patch_size, bias=bias)

    def forward(self, tokens):
        patches = self.proj(tokens)        # (B, H, W, C*p*p)
        patches = patches.permute(0, 3, 1, 2)  # (B, C*p*p, nH, nW)

        H = patches.shape[2] * self.patch_size
        W = patches.shape[3] * self.patch_size
        
        x = F.fold(patches, output_size=(H, W),
                   kernel_size=self.patch_size, stride=self.patch_size, padding=0
        )                                  # (B, C, H, W)
        return x
