import imageio
import numpy as np
import matplotlib.pyplot as plt
import skfmm

import torch

from .util import *
from .resampling import Resampling
from .interpolation import Smoothstep

class SDF:
    @classmethod
    def compute(cls, 
        im:torch.Tensor, 
        tiling:bool=True, 
        length_factor:float=0.1, 
        threshold=0.) -> torch.Tensor:
        H, W = im.shape[1:]
        if threshold > 0.: im = (im > 0.5)
        if tiling: im = Resampler.tile(im, (H*3, W*3))
        im = im[0,...].numpy()
        im_norm = im * 2. - 1.
        lsq = (H*3)**2 + (W*3)**2 if tiling else H**2 + W**2
        max_dist = np.sqrt(lsq) * length_factor
        distance = (skfmm.distance(im_norm) * 0.5 + (max_dist * 0.5)) / max_dist
        if tiling: distance = distance[H:H*2,W:W*2]
        return torch.from_numpy(distance).unsqueeze(0)

    @classmethod
    def render(cls, 
        sdf:torch.Tensor, 
        shape=tuple, 
        edge0:float=0.496, 
        edge1:float=0.498, 
        value0:float=0., 
        value1:float=1.,
        interpolant='quintic_polynomial',
        mode:str='bilinear'):
        H, W = shape[0], shape[1]
        sdf = Resampler.resize(sdf, (H, W), mode=mode)
        ones = torch.ones_like(sdf)
        interp = Smoothstep.apply(interpolant, edge0, edge1, sdf)
        render = torch.lerp(ones * value0, ones * value1, interp)
        return render