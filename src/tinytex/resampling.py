from __future__ import annotations
import typing
from typing import Union

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

from .util import *

class Resampling:

    """Image resizing and padding."""

    err_size = "tensor must be sized [C, H, W] or [N, C, H, W]"

    @classmethod
    def tile(cls, im:torch.Tensor, shape:tuple) -> torch.Tensor:
        """
        Tile/repeat image tensor to match target shape.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W].
        :param shape: Target shape as (height, width) tuple.
        :return: Padded image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        new_H, new_W = shape[0], shape[1]
        H, W = im.shape[2:]
        if H == new_H and W == new_W: return im.squeeze(0) if nobatch else im
        assert H <= new_H and W <= new_W, "target shape cannot be smaller than input shape"
        h_tiles = (new_H // H) + 1
        w_tiles = (new_W // W) + 1
        tiled_tensor = im.repeat(1, 1, h_tiles, w_tiles)
        tiled_tensor = tiled_tensor[..., :new_H, :new_W]
        return tiled_tensor.squeeze(0) if nobatch else tiled_tensor

    @classmethod
    def tile_n(cls, im:torch.Tensor, repeat_h:int, repeat_w:int):
        """
        Tile/repeat image tensor by number of repetitions.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W]
        :param repeat_h: Number of times to repeat image vertically.
        :param repeat_w: Number of times to repeat image horizontally.
        :return: Padded image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)        
        H, W = im.shape[2:]
        tiled_tensor = cls.tile(im, shape=(H*repeat_h, W*repeat_w))
        return tiled_tensor.squeeze(0) if nobatch else tiled_tensor

    @classmethod
    def tile_to_square(cls, im:torch.Tensor, target_size:int) -> torch.Tensor:
        """
        Tile image tensor to square dimensions of target size.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W].
        :return: Padded image tensor sized [C, H, W] or [N, C, H, W] where H = W.
        """
        # Uses numpy for legacy reasons, but can be reworked to use torch tile method above.
        # F.pad won't tile the image to arbitrary size.
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        N, C, H, W = im.size()
        if H == new_H and W == new_W: return im.squeeze(0) if nobatch else im
        assert H <= target_size and W <= target_size, "target shape cannot be smaller than input shape"
        np_image = im.permute(0, 2, 3, 1).numpy()
        tiled_image = np.ndarray([N, target_size, target_size, C])
        for i in range(im.shape[0]):
            h_tiles = int(np.ceil(target_size / np_image[i].shape[1]))
            v_tiles = int(np.ceil(target_size / np_image[i].shape[0]))
            tiled_image[i] = np.tile(np_image[i], (v_tiles, h_tiles, 1))[:target_size, :target_size, :]
        res = torch.from_numpy(tiled_image).permute(0, 3, 1, 2)
        return res.squeeze(0) if nobatch else res

    @classmethod
    def crop(cls, im:torch.Tensor, shape:tuple, start:tuple=(0, 0)):
        """
        Crop image tensor to maximum target shape, if and only if a crop box target dimension 
        is smaller than the boxed image dimension. Returned tensor can be smaller than target 
        shape, depending on input image shape - i.e. no automatic padding.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W].
        :param shape: Target shape as (height, width) tuple.
        :param start: Top-left corner coordinates of the crop box as (top, left) tuple.
        :return: Cropped image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        assert start[0] < im.size(2) and start[1] < im.size(2), \
            "crop box start dimensions must be smaller than image dimensions"
        H, W = min(im.size(2) - start[0], shape[0]), min(im.size(3) - start[1], shape[1])
        if H == im.size(2) and W == im.size(3): return im.squeeze(0) if nobatch else im
        res = TF.crop(im, top=start[0], left=start[1], height=H, width=W)
        return res.squeeze(0) if nobatch else res

    @classmethod
    def resize(cls, im:torch.Tensor, shape:tuple, mode:str='bilinear', iterative_downsample=False):    
        """
        Resize image tensor to target shape.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W].
        :param shape: Target shape as (height, width) tuple.
        :param mode: Resampleing algorithm ('nearest' | 'linear' | 'bilinear' | 'bicubic' | 'area').
        :param iterative_downsample: Iteratively average pixels if image dimension must be reduced 2x or more.
        :return: Resampled image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        res = None

        if iterative_downsample:
            # Use symmetric pooling until both dimensions are < 2x target.
            while im.size(2) >= shape[1] * 2 and im.size(3) >= shape[0] * 2:
                im = F.avg_pool2d(im, kernel_size=2, stride=2)

        res = F.interpolate(im, size=shape, mode=mode, align_corners=False)
        return res.squeeze(0) if nobatch else res

    @classmethod
    def resize_se(cls, im:torch.Tensor, size:int, mode:str='bilinear', iterative_downsample=False) -> torch.Tensor:
        """
        Resize image tensor by shortest edge, constraining proportions.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W]
        :param size: Target size for shortest edge
        :param mode: Resampleing algorithm ('nearest' | 'linear' | 'bilinear' | 'bicubic' | 'area')
        :param iterative_downsample: Iteratively average pixels if image dimension must be reduced 2x or more.
        :return: Resampled image tensor sized [C, H, W] or [N, C, H, W]
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        H, W = im.shape[2:]
        if min(H, W) == size: return im.squeeze(0) if nobatch else im
        scale = size / min(H, W)
        new_h = int(np.ceil(H * scale))
        new_w = int(np.ceil(W * scale))

        if iterative_downsample:
            # Use symmetric pooling until both dimensions are < 2x target.
            while im.size(2) >= new_h * 2 and im.size(3) >= new_w * 2:
                im = F.avg_pool2d(im, kernel_size=2, stride=2)

        
        res = F.interpolate(im, size=(new_h, new_w), mode=mode, align_corners=False)
        return res.squeeze(0) if nobatch else res

    @classmethod
    def resize_le(cls, im:torch.Tensor, size:int, mode:str='bilinear', iterative_downsample=False) -> torch.Tensor:
        """
        Resize image tensor by longest edge, constraining proportions.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W]
        :param size: Target size for longest edge
        :param mode: Resampleing algorithm ('nearest' | 'linear' | 'bilinear' | 'bicubic' | 'area')
        :param iterative_downsample: Iteratively average pixels if image dimension must be reduced 2x or more.
        :return: Resampled image tensor sized [C, H, W] or [N, C, H, W]
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        H, W = im.shape[2:]
        if max(H, W) == size: return im.squeeze(0) if nobatch else im
        scale = size / max(H, W)
        new_h = int(np.ceil(H * scale))
        new_w = int(np.ceil(W * scale))

        if iterative_downsample:
            # Use symmetric pooling until both dimensions are < 2x target.
            while im.size(2) >= new_h * 2 and im.size(3) >= new_w * 2:
                im = F.avg_pool2d(im, kernel_size=2, stride=2)

        
        res = F.interpolate(im, size=(new_h, new_w), mode=mode, align_corners=False)
        return res.squeeze(0) if nobatch else res

    def resize_le_to_next_pot(im:torch.Tensor, mode:str='bilinear'):
        """
        Resize image tensor by longest edge up to next higher power-of-two, constraining proportions.

        :param torch.tensor im: Image tensor sized [C, H, W] or [N, C, H, W].
        :return: Resampled image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        H, W = im.shape[2:]
        max_dim = max(H, W)
        size = next_pot(max_dim)
        res = cls.resize_le(im, size=size, mode=mode)
        return res.squeeze(0) if nobatch else res

    @classmethod
    def pad_rb(cls, im:torch.Tensor, shape:tuple, mode:str='replicate') -> torch.Tensor:
        """
        Pad image tensor to the right and bottom to target shape.

        :param torch.Tensor im: Image tensor sized [C, H, W] or [N, C, H, W].
        :param mode: Padding algorithm ('constant' | 'reflect' | 'replicate' | 'circular').
        :return: Padded image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)

        H, W = im.shape[2:]
        th, tw = shape[0], shape[1]
        assert tw > W and th > H, "target dimensions must be larger than image dimensions"

        pad_h = int(th - H)
        pad_w = int(tw - W)
        padding = (0, pad_w, 0, pad_h)
        if (H < th / 2. or W < tw / 2.) and (mode == "circular" or mode == "reflect"): 
            if mode == "reflect": raise Exception("target size too big for reflect mode")
            # Can't use torch pad, because dimensions won't allow it - fall back to manual repeat.
            padded_image = cls.tile(im.cpu(), shape=shape).to(im.device)
        else:
            padded_image = F.pad(im, padding, mode=mode, value=0)

        if nobatch: padded_image = padded_image.squeeze(0)
        return padded_image

    @classmethod
    def pad_to_next_pot(cls, im:torch.Tensor, mode:str='replicate') -> torch.Tensor:
        """
        Pad image tensor to next highest power-of-two square dimensions.

        :param torch.Tensor im: image tensor sized [C, H, W] or [N, C, H, W]
        :param mode: padding algorithm ('constant' | 'reflect' | 'replicate' | 'circular')
        :return: padded image tensor sized [C, H, W] or [N, C, H, W] where H = W
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        H, W = im.shape[2:]
        size = next_pot(max(H, W))
        padded_image = cls.pad_rb(im, shape=(size, size), mode=mode)
        if nobatch: padded_image = padded_image.squeeze(0)
        return padded_image

    
    @classmethod
    def generate_mip_pyramid(cls, im: Union[torch.Tensor]) -> torch.Tensor:
        """
        Generate a mipmap pyramid from input image and store it in a single image tensor.
        Pyramid is stored by placing the base image in the left portion and stacking
        each downsampled mip level vertically in the right portion.
        
        :param im: Input image tensor shape [C, H, W]
        :return: Pyramid tensor of shape [C, H, W + W//2]
        """
        C, H, W = im.size()

        pyramid = torch.zeros(C, H, W + W // 2, dtype=im.dtype, device=im.device)

        pyramid[:, :H, :W] = im

        HO = 0        
        mip = im.clone()
        max_mip = H.bit_length() - 1

        for i in range(1, max_mip + 1):
            NH, NW = H >> i, max(W >> i, 1)
            mip = F.avg_pool2d(mip, kernel_size=2, stride=2)
            pyramid[:, HO:HO+NH, W:W+NW] = mip
            HO += NH

        return pyramid

    @classmethod
    def compute_lod_offsets(cls, height: int) -> List[int]:
        """
        Compute the vertical offsets for each mip level in a mip pyramid.
        
        :param height: Height of the base image, determines the number of mip levels and their vertical offsets.
        :return: List of vertical offsets (in pixels) for each mip level. The base level is not included.
        """
        max_mip = height.bit_length() - 1
        offsets = []
        ho = 0
        for i in range(1, max_mip + 1):
            nh = height >> i  # Height of the current mip level
            offsets.append(ho)
            ho += nh
        return offsets

    @classmethod
    def sample_lod_bilinear(cls, pyramid: torch.Tensor, height: int, width: int, lod: float):
        """
        Resample a mip pyramid using standard bilinear interpolation at a given LOD.

        :param pyramid: Input mip pyramid tensor of shape [C, H, W], with stacked mips along height.
        :param height: Output image height.
        :param width: Output image width.
        :param lod: Level-of-detail value to sample from (can be fractional).
        :return: Resampled image tensor of shape [C, height, width].
        """
        C, H_total, W_total = pyramid.shape
        offsets = cls.compute_lod_offsets(H_total)
        lod_floor = int(lod)
        lod_ceil = min(lod_floor + 1, len(offsets) - 1)
        alpha = lod - lod_floor 

        W0_start = int(W_total // 3 * 2) if lod_floor > 0 else 0
        W1_start = int(W_total // 3 * 2) if lod_ceil > 0 else 0

        H0, W0 = max(1, H_total >> lod_floor), max(1, W0_start >> lod_floor)
        H1, W1 = max(1, H_total >> lod_ceil), max(1, W1_start >> lod_ceil)
        
        O0 = offsets[lod_floor-1] if lod_floor > 0 else 0
        O1 = offsets[lod_ceil-1] if lod_ceil > 0 else 0

        mip0 = pyramid[:, O0 : O0 + H0, W0_start : W0_start + W0]
        mip1 = pyramid[:, O1 : O1 + H1, W1_start : W1_start + W1]

        mip0_resized = F.interpolate(mip0.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
        mip1_resized = F.interpolate(mip1.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)

        return torch.lerp(mip0_resized, mip1_resized, alpha)

    @classmethod
    def sample_lod_bspline_hybrid(cls, pyramid: torch.Tensor, height: int, width: int, lod: float):
        """
        Resample a mip pyramid using a hybrid (4-tap bilinear) B-spline filter approximation.

        :param pyramid: Input mip pyramid tensor of shape [C, H, W], with stacked mips along height.
        :param height: Output image height.
        :param width: Output image width.
        :param lod: Level-of-detail value to sample from (can be fractional).
        :return: Resampled image tensor of shape [C, height, width].
        """
        def sample_level(mip, target_h, target_w):
            C_mip, H_mip, W_mip = mip.shape
            device = mip.device

            u_uv = torch.linspace(0, 1 - 1e-6, target_w, device=device)
            v_uv = torch.linspace(0, 1 - 1e-6, target_h, device=device)
            gy, gx = torch.meshgrid(v_uv, u_uv, indexing='ij')  # (H, W)

            u = gx * W_mip
            v = gy * H_mip

            t_x = u - torch.floor(u)
            t_y = v - torch.floor(v)
            f2_x, f3_x = t_x**2, t_x**3
            f2_y, f3_y = t_y**2, t_y**3

            wx0 = f2_x - 0.5*(f3_x + t_x)
            wx1 = 1.5*f3_x - 2.5*f2_x + 1.0
            wx2 = -1.5*f3_x + 2*f2_x + 0.5*t_x
            wx3 = 0.5*(f3_x - f2_x)
            sx0, sx1 = wx0 + wx1, wx2 + wx3
            fx0 = wx1 / (sx0 + 1e-8)
            fx1 = wx3 / (sx1 + 1e-8)

            wy0 = f2_y - 0.5*(f3_y + t_y)
            wy1 = 1.5*f3_y - 2.5*f2_y + 1.0
            wy2 = -1.5*f3_y + 2*f2_y + 0.5*t_y
            wy3 = 0.5*(f3_y - f2_y)
            sy0, sy1 = wy0 + wy1, wy2 + wy3
            fy0 = wy1 / (sy0 + 1e-8)
            fy1 = wy3 / (sy1 + 1e-8)

            base_x = torch.floor(u)
            base_y = torch.floor(v)

            t0_x = torch.clamp(base_x - 0.5 - 1 + fx0, 0, W_mip-1)
            t1_x = torch.clamp(base_x - 0.5 + 1 + fx1, 0, W_mip-1)

            t0_y = torch.clamp(base_y - 0.5 - 1 + fy0, 0, H_mip-1)
            t1_y = torch.clamp(base_y - 0.5 + 1 + fy1, 0, H_mip-1)

            def make_grid(x, y):
                return torch.stack([
                    ((x + 0.5) / W_mip * 2 - 1), 
                    ((y + 0.5) / H_mip * 2 - 1)
                ], dim=-1)

            grids = [
                make_grid(t0_x, t0_y),
                make_grid(t0_x, t1_y),
                make_grid(t1_x, t0_y),
                make_grid(t1_x, t1_y)
            ]

            samples = []
            for grid in grids:
                sampled = F.grid_sample(
                    mip.unsqueeze(0),
                    grid.unsqueeze(0),
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False  # must match texel center calculation
                )
                samples.append(sampled.squeeze(0))

            samples_tensor = torch.stack(samples, dim=0)  # (4, C, H, W)
            weights = torch.stack([
                sx0 * sy0,
                sx0 * sy1,
                sx1 * sy0,
                sx1 * sy1
            ], dim=0).unsqueeze(1)  # (4, 1, H, W)

            return (samples_tensor * weights).sum(dim=0)

        C, H_total, W_total = pyramid.shape
        offsets = cls.compute_lod_offsets(H_total)
        lod_floor = int(lod)
        lod_ceil = min(lod_floor + 1, len(offsets) - 1)
        alpha = lod - lod_floor

        if lod_floor > 0:
            W0_start = W_total // 3 * 2
            H0 = max(1, H_total >> lod_floor)
            W0 = max(1, W0_start >> lod_floor)
            O0 = offsets[lod_floor-1]
        else:
            W0_start, H0, W0, O0 = 0, H_total, W_total, 0
        mip0 = pyramid[:, O0:O0+H0, W0_start:W0_start+W0]

        if lod_ceil > 0:
            W1_start = W_total // 3 * 2
            H1 = max(1, H_total >> lod_ceil)
            W1 = max(1, W1_start >> lod_ceil)
            O1 = offsets[lod_ceil-1]
        else:
            W1_start, H1, W1, O1 = 0, H_total, W_total, 0
        mip1 = pyramid[:, O1:O1+H1, W1_start:W1_start+W1]

        sampled0 = sample_level(mip0, height, width)
        sampled1 = sample_level(mip1, height, width)
        return torch.lerp(sampled0, sampled1, alpha)

    @classmethod
    def sample_lod_bspline_dither(pyramid: torch.Tensor, height: int, width: int, lod: float):
        """
        Resample a mip pyramid using stochastic B-spline dithering.

        Converts the B-spline weights into a probability distribution and samples values probabilistically.

        :param pyramid: Input mip pyramid tensor of shape [C, H, W], with stacked mips along height.
        :param height: Output image height.
        :param width: Output image width.
        :param lod: Level-of-detail value to sample from (can be fractional).
        :return: Resampled image tensor of shape [C, height, width].
        """
        def sample_level(mip, target_h, target_w):
            C_mip, H_mip, W_mip = mip.shape
            device = mip.device

            x = torch.linspace(0.0, W_mip - 1.0, target_w, device=device)
            y = torch.linspace(0.0, H_mip - 1.0, target_h, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (H, W)

            floor_x = torch.floor(grid_x)  
            floor_y = torch.floor(grid_y)

            tx = grid_x - (floor_x)  
            ty = grid_y - (floor_y)

            w0_x = 1 + tx*(-3 + tx*(3 + tx*(-1)))
            w1_x = 5 + tx*(-3 + tx*(-3 + tx*2))
            w2_x = 6 - tx**3
            w0_y = 1 + ty*(-3 + ty*(3 + ty*(-1)))
            w1_y = 5 + ty*(-3 + ty*(-3 + ty*2))
            w2_y = 6 - ty**3

            xi = torch.rand((target_h, target_w, 2), device=device)
            rx = xi[..., 0] * 6
            ry = xi[..., 1] * 6

            cum_x = torch.stack([w0_x, w0_x + w1_x, w0_x + w1_x + w2_x], dim=-1)
            mask_x = (rx.unsqueeze(-1) < cum_x).int().argmax(dim=-1)
            offset_x = mask_x - 1

            cum_y = torch.stack([w0_y, w0_y + w1_y, w0_y + w1_y + w2_y], dim=-1)
            mask_y = (ry.unsqueeze(-1) < cum_y).int().argmax(dim=-1)
            offset_y = mask_y - 1

            tap_x = torch.clamp(floor_x + offset_x, 0, W_mip-1)
            tap_y = torch.clamp(floor_y + offset_y, 0, H_mip-1)

            norm_x = (tap_x / (W_mip-1)) * 2 - 1
            norm_y = (tap_y / (H_mip-1)) * 2 - 1
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

            return F.grid_sample(
                mip.unsqueeze(0),
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            ).squeeze(0)

        C, H_total, W_total = pyramid.shape    
        offsets = cls.compute_lod_offsets(H_total)
        lod_floor = int(lod)
        lod_ceil = min(lod_floor + 1, len(offsets) - 1)
        alpha = lod - lod_floor

        if lod_floor > 0:
            W0_start = W_total // 3 * 2
            H0 = max(1, H_total >> lod_floor)
            W0 = max(1, W0_start >> lod_floor)
            O0 = offsets[lod_floor-1]
        else:
            W0_start, H0, W0, O0 = 0, H_total, W_total, 0
        mip0 = pyramid[:, O0:O0+H0, W0_start:W0_start+W0]

        if lod_ceil > 0:
            W1_start = W_total // 3 * 2
            H1 = max(1, H_total >> lod_ceil)
            W1 = max(1, W1_start >> lod_ceil)
            O1 = offsets[lod_ceil-1]
        else:
            W1_start, H1, W1, O1 = 0, H_total, W_total, 0
        mip1 = pyramid[:, O1:O1+H1, W1_start:W1_start+W1]

        sampled0 = sample_level(mip0, height, width)
        sampled1 = sample_level(mip1, height, width)
        return torch.lerp(sampled0, sampled1, alpha)