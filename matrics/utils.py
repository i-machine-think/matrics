import numpy as np


def one_hot(a):
    """
    Returns one hot encoding of passed numpy array
    Args:
        a (np.array)
    """
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out
