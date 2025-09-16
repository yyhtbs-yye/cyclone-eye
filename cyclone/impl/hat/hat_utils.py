from functools import lru_cache
import torch

@lru_cache(maxsize=10)
def calculate_rpi_oca(window_size, overlap_ratio):
    """
    Calculate relative position index for overlap cross-attention when 
    window_size is (H, W) and extend_window_size is (He, We), where
    He = H * (1 + overlap_ratio) and We = W * (1 + overlap_ratio).
    Returns: [H*W, He*We] LongTensor of linearized 2D relative positions.
    """

    # window_size: tuple (H, W), overlap_ratio: float
    H, W = window_size
    He = H + int(overlap_ratio * H)
    We = W + int(overlap_ratio * W)

    # original window coords
    coords_h = torch.arange(H)
    coords_w = torch.arange(W)
    coords_ori = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, H, W
    coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, H*W

    # extended window coords
    coords_h_e = torch.arange(He)
    coords_w_e = torch.arange(We)
    coords_ext = torch.stack(torch.meshgrid(coords_h_e, coords_w_e, indexing='ij'))  # 2, He, We
    coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, He*We

    # relative coords: 2, H*W, He*We
    relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]

    # ws*ws, wse*wse, 2
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()

    # shift to start from 0 (match original formula per-dimension)
    shift_h = H - He + 1
    shift_w = W - We + 1
    relative_coords[:, :, 0] += shift_h
    relative_coords[:, :, 1] += shift_w

    # map (dh, dw) -> single index with per-dimension bases
    size_h = H + He - 1
    size_w = W + We - 1
    relative_coords[:, :, 0] *= size_w
    relative_position_index = relative_coords.sum(-1)
    return relative_position_index
