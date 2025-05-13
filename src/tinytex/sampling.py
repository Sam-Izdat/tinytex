from __future__ import annotations
import typing
from typing import Union

import torch
import torch.nn.functional as F

def generate_mip_pyramid(im: Union[torch.Tensor]) -> torch.Tensor:
    """
    Generate a mipmap pyramid from input image and store it in a single image tensor.
    Pyramid is stored by placing the base image in the left portion and stacking
    each downsampled mip level vertically in the right portion.
    
    :param im: Input image tensor shape [C, H, W]
    :return: Tuple containing:
             - mip pyramid tensor of shape [C, H, W + W//2]
             - list of vertical offsets (in pixels) for each mip level stored in the right portion.
               Base level is not included in this list.
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

def compute_mip_pyramid_offsets(H: int) -> List[int]:
    """
    Compute the vertical offsets for each mip level in a mip pyramid.
    
    :param H: Height of the base image, determines the number of mip levels and their vertical offsets.
    :return: List of vertical offsets (in pixels) for each mip level. The base level is not included.
    """
    max_mip = H.bit_length() - 1
    offsets = []
    ho = 0
    for i in range(1, max_mip + 1):
        nh = H >> i  # Height of the current mip level
        offsets.append(ho)
        ho += nh
    return offsets

def sample_mip_pyramid_bilinear(height, width, pyramid, lod):
    C, H_total, W_total = pyramid.shape
    offsets = compute_mip_pyramid_offsets(H_total)
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

def sample_mip_pyramid_bspline_hybrid(height, width, pyramid, lod):
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
    offsets = compute_mip_pyramid_offsets(H_total)
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

def sample_mip_pyramid_bspline_dither(height, width, pyramid, lod):
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
    offsets = compute_mip_pyramid_offsets(H_total)
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