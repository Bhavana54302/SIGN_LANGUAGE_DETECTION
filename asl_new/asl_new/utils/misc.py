import errno
import functools
import getpass
import os
import pickle as pkl
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io
import torch


def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


@functools.lru_cache(maxsize=64, typed=False)
def load_checkpoint(ckpt_path):
    return torch.load(ckpt_path, map_location={"cuda:0": "cpu"})


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(f"Cannot convert {type(ndarray)} to torch tensor")
    return ndarray