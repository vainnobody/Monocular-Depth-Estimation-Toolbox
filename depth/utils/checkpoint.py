import inspect
import os
from typing import Any

import torch


def torch_load_checkpoint(path: Any, map_location: str = 'cpu'):
    """Load a checkpoint with low-peak-memory options when available."""
    load_kwargs = dict(map_location=map_location)
    signature = inspect.signature(torch.load)

    if 'mmap' in signature.parameters and isinstance(path, (str, os.PathLike)):
        load_kwargs['mmap'] = True

    try:
        return torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop('mmap', None)
        return torch.load(path, **load_kwargs)


def load_state_dict_low_mem(module, state_dict, strict=False, assign=False):
    """Use native torch loading and opt into assign mode when supported."""
    load_kwargs = dict(strict=strict)
    signature = inspect.signature(module.load_state_dict)
    if assign and 'assign' in signature.parameters:
        load_kwargs['assign'] = True
    return module.load_state_dict(state_dict, **load_kwargs)
