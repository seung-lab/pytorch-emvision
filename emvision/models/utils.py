import collections.abc
import operator
from itertools import repeat

import torch


def _ntuple(n):
    """
    Copied from the PyTorch source code (https://github.com/pytorch).
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_triple = _ntuple(3)


def sum3(x,y):
    return (x[0]+y[0], x[1]+y[1], x[2]+y[2])


def mul3(x,y):
    return (x[0]*y[0], x[1]*y[1], x[2]*y[2])


def div3(x,y):
    return (x[0]//y[0], x[1]//y[1], x[2]//y[2])


def sub3(x,y):
    return (x[0]-y[0], x[1]-y[1], x[2]-y[2])


def residual_sum(x, skip, margin, residual):
    return x + crop3d(skip, margin) if residual else x


def crop3d(x, margin):
    shape = x.size()
    index3d = tuple(slice(b,e-b) for (b,e) in zip(margin,shape[-3:]))
    index = tuple(slice(0,e) for e in shape[:-3]) + index3d
    return x[index]


def crop3d_center(x, ref):
    xs = x.size()[-3:]
    rs = ref.size()[-3:]
    assert all(x >= r for x,r in zip(xs,rs))
    assert all((x - r) % 2 == 0 for x,r in zip(xs,rs))
    margin = [(x - r) // 2 for x,r in zip(xs,rs)]
    return crop3d(x, margin)


def pad3d_center(x, ref):
    xs = x.size()[-3:]
    rs = ref.size()[-3:]
    assert all(x <= r for x,r in zip(xs,rs))
    assert all((r - x) % 2 == 0 for x,r in zip(xs,rs))
    margin = [(r - x) // 2 for x,r in zip(xs,rs)]

    padded = torch.zeros_like(ref, dtype=x.dtype)
    padded[..., 
        margin[0]:-margin[0], margin[1]:-margin[1], margin[2]:-margin[2]
    ] = x

    return padded


def pad_size(kernel_size, mode):
    assert mode in ['valid', 'same', 'full']
    ks = _triple(kernel_size)
    if mode == 'valid':
        return _triple(0)
    elif mode == 'same':
        assert all(x % 2 for x in ks)
        return tuple(x // 2 for x in ks)
    elif mode == 'full':
        return tuple(x - 1 for x in ks)


def crop_margin(kernel_size, mode):
    """The margin to crop to match tensor sizes across convolutions."""
    if mode == "same":
        return _triple(0)
    elif mode == "valid":
        # margins == padding for "same" convolutions
        return pad_size(kernel_size, "same")
    else:
        raise ValueError(f"unsupported convolution mode: {mode}")
