from typing import Union, Tuple

import imageio
import numpy as np
import skfmm

import torch

from .util import *
from .resampling import Resampling
from .smoothstep import Smoothstep

class SDF:

    """Signed distance field computation and rendering."""

    err_tile = "cannot tile to dimensions smaller than SDF dimensions"

    @classmethod
    def compute(cls, 
        im: torch.Tensor, 
        periodic: bool = True,
        length_factor: float = 0.1, 
        threshold: Union[float, None] = None,
        tile_to: Union[tuple, None] = None
    ) -> torch.Tensor:
        """
        Compute a signed distance field from a binary image.

        :param im: Input tensor of shape [1, H, W].
        :param periodic: Whether to use periodic boundary conditions.
        :param length_factor: Scales the maximum measurable distance.
        :param threshold: Binarization threshold if `im` isn't binary.
        :param tile_to: Optional tiling target shape (H, W).
        :return: Tensor of shape [1, H, W] with values in [0, 1].
        """
        H, W = im.shape[1:]
        if tile_to is not None: assert max(H, W) <= tile_to, err_tile
        if threshold is not None: im = (im > threshold).float()
        im = im[0,...].numpy()
        im_norm = im * 2. - 1.
        distance = skfmm.distance(im_norm, periodic=periodic)
        distance = cls.__scale_dist(torch.from_numpy(distance), H, W, length_factor)
        out = distance.clamp(0.,1.).unsqueeze(0)
        if tile_to is not None: out = Resampling.tile(out, (tile_to, tile_to))
        return out

    @classmethod
    def render(cls,
        sdf: torch.Tensor,
        shape: tuple,
        edge0: float = 0.496,
        edge1: float = 0.498,
        value0: float = 0.,
        value1: float = 1.,
        interpolant: str = 'quintic_polynomial',
        mode: str = 'bilinear'
    ) -> torch.Tensor:
        """
        Render an SDF to a grayscale field using soft-thresholding.

        :param sdf: Input SDF tensor of shape [1, H, W].
        :param shape: Output size (height, width).
        :param edge0: Lower edge of the soft transition zone.
        :param edge1: Upper edge of the soft transition zone.
        :param value0: Output value below edge0.
        :param value1: Output value above edge1.
        :param interpolant: Interpolation curve used between edge0 and edge1.
        :param mode: Interpolation method for resizing.
        :return: Rendered tensor of shape [1, H, W].
        """
        H, W = shape[0], shape[1]
        sdf = Resampling.resize(sdf, (H, W), mode=mode)
        ones = torch.ones_like(sdf)
        interp = Smoothstep.apply(interpolant, edge0, edge1, sdf)
        render = torch.lerp(ones * value0, ones * value1, interp)
        return render

    @classmethod
    def min(cls, sdf1: torch.Tensor, sdf2: torch.Tensor) -> torch.Tensor:
        """Return the minimum of two SDFs (union)."""
        return torch.minimum(sdf1, sdf2)

    @classmethod
    def max(cls, sdf1: torch.Tensor, sdf2: torch.Tensor) -> torch.Tensor:
        """Return the maximum of two SDFs (intersection)."""
        return torch.maximum(sdf1, sdf2)

    @classmethod
    def circle(cls,
        size: int = 64,
        radius: int = 20,
        length_factor: float = 0.1,
        tile_to: Union[int, None] = None
    ) -> torch.Tensor:
        """
        Generate a circular SDF centered in the image.

        :param size: Output image size (square).
        :param radius: Radius of the circle.
        :param length_factor: Distance scaling factor.
        :param tile_to: Optional output tiling target.
        :return: SDF tensor of shape [1, size, size].
        """
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
        x_nrm, y_nrm = x - size // 2, y - size // 2
        dist = cls.__length(torch.stack([x_nrm, y_nrm], dim=-1)) - float(radius)
        out = cls.__scale_dist(dist, size, size, length_factor).unsqueeze(0).clamp(0., 1.)
        if tile_to is not None: 
            assert size <= tile_to, cls.err_tile
            out = Resampling.tile(out, (tile_to, tile_to))
        return out

    @classmethod
    def box(cls,
        size: int = 64,
        box_shape: tuple = (32, 32),
        length_factor: float = 0.1,
        tile_to: Union[int, None] = None
    ) -> torch.Tensor:
        """
        Generate a rectangular SDF centered in the image.

        :param size: Output image size (square).
        :param box_shape: (height, width) of the rectangle.
        :param length_factor: Distance scaling factor.
        :param tile_to: Optional output tiling target.
        :return: SDF tensor of shape [1, size, size].
        """
        if tile_to is not None: assert size <= tile_to, cls.err_tile
        h, w = box_shape[0] // 2, box_shape[1] // 2
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
        x_nrm, y_nrm = x - size // 2, y - size // 2
        hw = torch.tensor([h, w], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(size, size, 1)
        dist = torch.abs(torch.stack([x_nrm, y_nrm], dim=-1)) - hw
        dist = cls.__length(dist.clamp(0.)) + torch.minimum(
            torch.maximum(dist[..., 0], dist[..., 1]), torch.zeros_like(dist[..., 0])).squeeze(-1)
        out = cls.__scale_dist(dist, size, size, length_factor).unsqueeze(0).clamp(0., 1.)
        if tile_to is not None: out = Resampling.tile(out, (tile_to, tile_to))
        return out

    @classmethod
    def segment(cls,
        size: int = 64,
        a: tuple = (32, 0),
        b: tuple = (32, 64),
        length_factor: float = 0.1,
        tile_to: Union[int, None] = None
    ) -> torch.Tensor:
        """
        Generate an SDF for a finite line segment.

        :param size: Output image size (square).
        :param a: Starting point of the segment.
        :param b: Ending point of the segment.
        :param length_factor: Distance scaling factor.
        :param tile_to: Optional output tiling target.
        :return: SDF tensor of shape [1, size, size].
        """
        if tile_to is not None: assert size <= tile_to, cls.err_tile
        ax, ay = a
        bx, by = b
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
        x_nrm, y_nrm = x - size // 2, y - size // 2
        a = torch.tensor([ay - size // 2, ax - size // 2], dtype=torch.float32)
        b = torch.tensor([by - size // 2, bx - size // 2], dtype=torch.float32)
        pa = torch.stack([x_nrm, y_nrm], dim=-1) - a
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