from typing import Union, Tuple

import imageio
import numpy as np
import matplotlib.pyplot as plt
import skfmm

import torch

from .util import *
from .resampling import Resampling
from .interpolation import Smoothstep

class SDF:

    err_tile = "cannot tile to dimensions smaller than SDF dimensions"

    @classmethod
    def compute(cls, 
        im:torch.Tensor, 
        tiling:bool=True, 
        length_factor:float=0.1, 
        threshold=None,
        tile_to=None) -> torch.Tensor:
        H, W = im.shape[1:]
        if tile_to is not None: assert max(H, W) <= tile_to, err_tile
        if threshold is not None: im = (im > threshold)
        if tiling: im = Resampling.tile(im, (H*3, W*3))
        im = im[0,...].numpy()
        im_norm = im * 2. - 1.
        distance = skfmm.distance(im_norm)
        distance = cls.__scale_dist(torch.from_numpy(distance), (H*3 if tiling else H), (W*3 if tiling else W), length_factor)
        if tiling: distance = distance[H:H*2,W:W*2]
        out = distance.clamp(0.,1.).unsqueeze(0)
        if tile_to is not None: out = Resampling.tile(out, (tile_to, tile_to))
        return out

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
        sdf = Resampling.resize(sdf, (H, W), mode=mode)
        ones = torch.ones_like(sdf)
        interp = Smoothstep.apply(interpolant, edge0, edge1, sdf)
        render = torch.lerp(ones * value0, ones * value1, interp)
        return render

    @classmethod
    def min(cls, sdf1, sdf2):
        return torch.minimum(sdf1, sdf2)

    @classmethod
    def max(cls, sdf1, sdf2):
        return torch.maximum(sdf1, sdf2)

    @classmethod
    def circle(cls, 
        size:int=64, 
        radius:int=20, 
        length_factor:float=0.1, 
        tile_to:int=None) -> torch.Tensor:
        assert size <= tile_to, cls.err_tile
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
        x_ndc, y_ndc = x - size // 2, y - size // 2
        dist = cls.__length(torch.stack([x_ndc, y_ndc], dim=-1)) - float(radius)
        out = cls.__scale_dist(dist, size, size, length_factor).unsqueeze(0).clamp(0., 1.)
        if tile_to is not None: out = Resampling.tile(out, (tile_to, tile_to))
        return out

    @classmethod
    def box(cls, 
        size:int=64, 
        box_shape:Tuple[int, int]=(32, 32), 
        length_factor:float=0.1, 
        tile_to:int=None) -> torch.Tensor:
        assert size <= tile_to, cls.err_tile
        h, w = box_shape[0] // 2, box_shape[1] // 2
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
        x_ndc, y_ndc = x - size // 2, y - size // 2
        hw = torch.tensor([h, w], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(size, size, 1)
        dist = torch.abs(torch.stack([x_ndc, y_ndc], dim=-1)) - hw
        dist = cls.__length(dist.clamp(0.)) + torch.minimum(
            torch.maximum(dist[..., 0], dist[..., 1]), torch.zeros_like(dist[..., 0])).squeeze(-1)
        out = cls.__scale_dist(dist, size, size, length_factor).unsqueeze(0).clamp(0., 1.)
        if tile_to is not None: out = Resampling.tile(out, (tile_to, tile_to))
        return out

    @classmethod
    def segment(cls, 
        size:int=64, 
        a:Tuple[int, int]=(32, 0), 
        b:Tuple[int, int]=(32, 64), 
        length_factor:float=0.1, 
        tile_to:int=None) -> torch.Tensor:
        assert size <= tile_to, cls.err_tile
        ax, ay = a
        bx, by = b
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
        x_ndc, y_ndc = x - size // 2, y - size // 2
        a = torch.tensor([ay - size // 2, ax - size // 2], dtype=torch.float32)
        b = torch.tensor([by - size // 2, bx - size // 2], dtype=torch.float32)
        pa = torch.stack([x_ndc, y_ndc], dim=-1) - a
        ba = b - a
        h = torch.clamp(torch.sum(pa * ba, dim=-1) / torch.sum(ba * ba), 0.0, 1.0)
        dist = cls.__length(pa - ba.unsqueeze(0) * h.unsqueeze(-1))        
        out = cls.__scale_dist(dist, size, size, length_factor).unsqueeze(0).clamp(0., 1.)
        if tile_to is not None: out = Resampling.tile(out, (tile_to, tile_to))
        return out

    @classmethod
    def __length(cls, vec):
        return torch.norm(vec.float(), dim=-1)

    @classmethod
    def __scale_dist(cls, dist, h, w, fac=0.1):
        max_dist = np.sqrt(h**2 + w**2) * fac
        dist = (dist * 0.5 + (max_dist * 0.5)) / max_dist
        return dist.clamp(0., 1.)