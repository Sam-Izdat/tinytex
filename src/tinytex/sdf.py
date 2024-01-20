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
        if tiling: im = Resampling.tile(im, (H*3, W*3))
        im = im[0,...].numpy()
        im_norm = im * 2. - 1.
        # lsq = (H*3)**2 + (W*3)**2 if tiling else H**2 + W**2
        # max_dist = np.sqrt(lsq) * length_factor
        # distance = (skfmm.distance(im_norm) * 0.5 + (max_dist * 0.5)) / max_dist
        distance = skfmm.distance(im_norm)
        distance = cls.__scale_dist(distance, (H*3 if tiling else H), (W*3 if tiling else W), length_factor)
        if tiling: distance = distance[H:H*2,W:W*2]
        return torch.from_numpy(distance).clamp(0.,1.).unsqueeze(0)

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

    # @classmethod
    # def __scale_dist(cls, dist, h, w, fac=0.1):
    #     max_dist = np.sqrt(h**2 + w**2) * fac
    #     dist = (dist * 0.5 + (max_dist * 0.5)) / max_dist
    #     return dist.clip(0., 1.)

    # @classmethod
    # def __length(cls, vec):
    #     return np.linalg.norm(vec)

    @classmethod
    def __length(cls, vec):
        return torch.norm(vec.float(), dim=-1)

    @classmethod
    def __scale_dist(cls, dist, h, w, fac=0.1):
        max_dist = np.sqrt(h**2 + w**2) * fac
        dist = (dist * 0.5 + (max_dist * 0.5)) / max_dist
        return dist.clamp(0., 1.)

    @classmethod
    def circle(cls, size=256, radius=100., length_factor=0.1):
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
        x_ndc, y_ndc = x - size // 2, y - size // 2
        dist = cls.__length(torch.stack([x_ndc, y_ndc], dim=-1)) - radius
        out = cls.__scale_dist(dist, size, size, length_factor).unsqueeze(0)
        return out

    @classmethod
    def box(cls, size=256, h=64, w=64, length_factor=0.1):
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
        x_ndc, y_ndc = x - size // 2, y - size // 2
        hw = torch.tensor([h, w], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(size, size, 1)
        dist = torch.abs(torch.stack([x_ndc, y_ndc], dim=-1)) - hw
        dist = cls.__length(dist.clamp(0.)) + torch.minimum(
            torch.maximum(dist[..., 0], dist[..., 1]), torch.zeros_like(dist[..., 0])).squeeze(-1)
        out = cls.__scale_dist(dist, size, size, length_factor).unsqueeze(0)
        return out

    @classmethod
    def segment(cls, size=256, ax=128, ay=0, bx=128, by=256, length_factor=0.1):
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
        x_ndc, y_ndc = x - size // 2, y - size // 2
        a = torch.tensor([ay - size // 2, ax - size // 2], dtype=torch.float32)
        b = torch.tensor([by - size // 2, bx - size // 2], dtype=torch.float32)
        pa = torch.stack([x_ndc, y_ndc], dim=-1) - a
        ba = b - a
        h = torch.clamp(torch.sum(pa * ba, dim=-1) / torch.sum(ba * ba), 0.0, 1.0)
        dist = cls.__length(pa - ba.unsqueeze(0) * h.unsqueeze(-1))        
        out = cls.__scale_dist(dist, size, size, length_factor).unsqueeze(0)
        return out

    # @classmethod
    # def circle(cls, size=256, r=100.):
    #     out = np.ndarray([1, size, size])
    #     for x in range(size):
    #         for y in range(size):
    #             x_ndc, y_ndc = y - int(size/2.), x - int(size/2.)
    #             dist = cls.__length(np.array([x_ndc, y_ndc])) - r
    #             out[..., y, x] = cls.__scale_dist(dist, size, size)
    #     return torch.from_numpy(out)

    # @classmethod
    # def box(cls, size=256, h=64, w=64):
    #     out = np.ndarray([1, size, size])
    #     for x in range(size):
    #         for y in range(size):
    #             x_ndc, y_ndc = y - int(size/2.), x - int(size/2.)
    #             dist = np.abs(np.array([x_ndc, y_ndc]))-np.array([h, w]);
    #             dist = cls.__length(np.maximum(dist, 0.0)) + np.min(np.max(dist[0],dist[1]),0.0);
    #             out[..., y, x] = cls.__scale_dist(dist, size, size)
    #     return torch.from_numpy(out)

    # @classmethod
    # def segment(cls, size=256, ax=128, ay=0, bx=128, by=256):
    #     out = np.ndarray([1, size, size])
    #     a = np.array([ay - int(size/2.), ax - int(size/2.)])
    #     b = np.array([by - int(size/2.), bx - int(size/2.)])
    #     for x in range(size):
    #         for y in range(size):
    #             x_ndc, y_ndc = y - int(size/2.), x - int(size/2.)
    #             pa = np.array([x_ndc, y_ndc]) - a
    #             ba = b - a
    #             h = np.clip(np.dot(pa, ba)/np.dot(ba, ba), 0.0, 1.0 )
    #             dist = cls.__length(pa - ba * h)
    #             out[..., y, x] = cls.__scale_dist(dist, size, size)
    #     return torch.from_numpy(out)