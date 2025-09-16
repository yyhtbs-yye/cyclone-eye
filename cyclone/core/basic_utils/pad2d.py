import torch.nn.functional as F

def pad_by_win_2d(x, window_size, pad_mode='reflect'):
        
    H, W = x.shape[-2:]
    
    mod_pad_h = (window_size - H % window_size) % window_size
    mod_pad_w = (window_size - W % window_size) % window_size

    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), pad_mode)

    return x, mod_pad_w, mod_pad_h

def unpad_by_win_2d(x, mod_pad_w, mod_pad_h):
    """
    Inverse of pad_by_win_2d. Removes the rightmost `mod_pad_w` columns and
    the bottom `mod_pad_h` rows from the last two dims (H, W).

    Works for tensors shaped (..., H, W) or (N, C, H, W), etc.
    """
    if mod_pad_h and mod_pad_h > 0:
        x = x[..., :-mod_pad_h, :]  # crop bottom rows
    if mod_pad_w and mod_pad_w > 0:
        x = x[..., :, :-mod_pad_w]  # crop rightmost columns
    return x
