from einops import rearrange

def grid_shuffle(x, interval_size: int):
    """
    Split the H and W dimensions into grids of size `interval_size` and
    fold the s1*s2 positions into the batch dimension.

    Args:
        x: Tensor of shape (b, c, h, w)
        interval_size (int): grid interval size (must divide h and w)

    Returns:
        Tensor of shape (b * interval_size * interval_size, c, h // interval_size, w // interval_size)
    """
    b, c, h, w = x.shape
    s = interval_size
    assert h % s == 0 and w % s == 0, "h and w must be divisible by interval_size"

    return rearrange(
        x,
        "b c (hb s1) (wb s2) -> (b s1 s2) c hb wb",
        s1=s, s2=s
    )


def grid_unshuffle(x, interval_size: int):
    """
    Inverse of grid_shuffle: restore the original (b, c, h, w) tensor.

    Args:
        x: Tensor of shape (b * interval_size * interval_size, c, h // interval_size, w // interval_size)
        interval_size (int): grid interval size used in grid_shuffle

    Returns:
        Tensor of shape (b, c, h, w)
    """
    s = interval_size
    # We don't know original b directly; einops handles it symbolically.
    return rearrange(
        x,
        "(b s1 s2) c hb wb -> b c (hb s1) (wb s2)",
        s1=s, s2=s
    )
