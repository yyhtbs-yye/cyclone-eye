import torch

from functools import lru_cache

@lru_cache(maxsize=10)
def calculate_rpi_sa(window_size):
    """
    Calculate relative position index for self-attention when window_size is (Wh, Ww).
    Returns: [Wh*Ww, Wh*Ww] LongTensor of linearized 2D relative positions.
    """
    Wh, Ww = window_size  # height, width
    coords_h = torch.arange(Wh)
    coords_w = torch.arange(Ww)
    # 2, Wh, Ww  (use 'ij' to avoid default-change warning)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    # 2, N, N  where N = Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    # N, N, 2
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()

    # shift to start from 0
    relative_coords[:, :, 0] += Wh - 1  # delta_h in [0, 2*Wh-2]
    relative_coords[:, :, 1] += Ww - 1  # delta_w in [0, 2*Ww-2]

    # map 2D offsets -> 1D index
    # scale height offsets by width range
    relative_coords[:, :, 0] *= (2 * Ww - 1)
    relative_position_index = relative_coords.sum(-1)  # [N, N]
    return relative_position_index
