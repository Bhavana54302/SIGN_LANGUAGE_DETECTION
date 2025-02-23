import random

import numpy as np
import torch

from .imutils import im_to_video, video_to_im


def color_normalize(x, mean, std):
    """Normalize a tensor of images by subtracting (resp. dividing) by the mean (resp.
    std. deviation) statistics of a dataset in RGB space.
    """
    if x.dim() in {3, 4}:
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        assert x.size(0) == 3, "For single video format, expected RGB along first dim"
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
    elif x.dim() == 5:
        assert (
            x.shape[1] == 3
        ), "For batched video format, expected RGB along second dim"
        x[:, 0].sub_(mean[0]).div_(std[0])
        x[:, 1].sub_(mean[1]).div_(std[1])
        x[:, 2].sub_(mean[2]).div_(std[2])
    return x