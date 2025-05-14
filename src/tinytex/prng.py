from __future__ import annotations
import typing
from typing import Union
import torch
import numpy as np

class PRNG:
    """Pseudo-random number generation."""

    # See: 
    # - FordPerfect int hash https://www.shadertoy.com/view/dllSW7
    # - MurmurHash https://github.com/aappleby/smhasher/tree/master/src

    @classmethod
    def pt_hash_uint(cls, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        Hash a 1D tensor of integers using a uint-style hash.

        :param x: Input int tensor.
        :param normalize: If True, returns float32 values in [0, 1].
        :return: Hashed tensor (uint32 or float32).
        """
        # Ensure x is a 64-bit integer and simulate uint32 by masking
        x = (x.to(torch.int64)) & 0xFFFFFFFF
        x ^= x >> 15
        x ^= ((x * x) | 1) & 0xFFFFFFFF  # Keep the value within uint32 bounds
        x ^= x >> 17
        x = (x * 0x9E3779B9) & 0xFFFFFFFF  # Multiply and wrap with uint32
        x ^= x >> 13
        r = (x & 0xFFFFFFFF) # Ensure final wraparound to uint32
        if normalize: r = r.to(torch.float32) / 0xFFFFFFFF
        return r

    @classmethod
    def pt_hash2_uint(cls, x: torch.Tensor, y: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        2D hash using XQO-style bit-mixing.

        :param x: X coordinates (int tensor).
        :param y: Y coordinates (int tensor).
        :param normalize: If True, returns float32 in [0, 1].
        :return: Hashed values (uint32 or float32).
        """
        x = (x.to(torch.int64)) & 0xFFFFFFFF
        y = (y.to(torch.int64)) & 0xFFFFFFFF
        h = pt_hash_uint((x * 0x85EBCA6B) ^ (y * 0xC2B2AE35))
        return h.to(torch.float32) / 0xFFFFFFFF if normalize else h

    @classmethod
    def pt_hash2_uv(cls, uv: torch.Tensor, seed: int = 0, tile_size: int = 1023) -> torch.Tensor:
        """
        Hash 2D normalized UV coordinates. Produces uniform pseudo-random values.

        :param uv: Tensor of shape [2, H, W] with values in [0, 1].
        :param seed: Optional seed for scrambling.
        :param tile_size: Maximum tile size (default: 1023).
        :return: Tensor of shape [H, W] with float32 values in [0, 1].
        """
        canvas_size = torch.tensor([uv.size(1), uv.size(2)], dtype=torch.float32, device=uv.device).view(2, 1, 1)
        coords = torch.floor((uv * canvas_size) % tile_size).to(torch.int64)  # int coords
        idx = coords[1] * uv.size(1) + coords[0]  # flattened index
        if seed != 0: idx = (idx ^ ((seed & 0xFFFF) << 16) ^ (seed * 0x27d4eb2d)) & 0xFFFFFFFF
        hashed_value = pt_hash_uint(idx)
        return hashed_value.to(torch.float32) / 0xFFFFFFFF

    @classmethod
    def pt_hash2_xy(cls, xy: torch.Tensor, seed: int = 0, tile_size: int = 1023) -> torch.Tensor:
        """
        Hash 2D XY integer coordinates. Produces uniform pseudo-random values.

        :param xy: Tensor of shape [2, H, W] with integer coordinates.
        :param seed: Optional seed for scrambling.
        :param tile_size: Maximum tile size (default: 1023).
        :return: Tensor of shape [H, W] with float32 values in [0, 1].
        """
        coords = torch.floor((xy) % tile_size).to(torch.int64)  # int coords
        idx = coords[1] * xy.size(1) + coords[0]  # flattened index
        if seed != 0: idx = (idx ^ ((seed & 0xFFFF) << 16) ^ (seed * 0x27d4eb2d)) & 0xFFFFFFFF
        hashed_value = pt_hash_uint(idx)
        return hashed_value.to(torch.float32) / 0xFFFFFFFF

    @classmethod
    def np_hash_uint(cls, x: np.ndarray, normalize: bool = False) -> np.ndarray:
        """
        1D NumPy-style uint hash.

        :param x: Input array (uint32-compatible).
        :param normalize: If True, returns float32 values in [0, 1].
        :return: Hashed values (uint32 or float32).
        """
        x = x.astype(np.uint32)
        with np.errstate(over='ignore'):
            x ^= x >> 15
            x ^= (x * x) | np.uint32(1)
            x ^= x >> 17
            x *= np.uint32(0x9E3779B9)
            x ^= x >> 13
        return x / np.float32(0xFFFFFFFF) if normalize else x

    @classmethod
    def np_hash2_uint(cls, x: np.ndarray, y: np.ndarray, normalize = False) -> np.ndarray:
        """
        2D NumPy-style uint hash using mixing constants.

        :param x: X array (uint32-compatible).
        :param y: Y array (uint32-compatible).
        :param normalize: If True, returns float32 values in [0, 1].
        :return: Hashed values (uint32 or float32).
        """
        with np.errstate(over='ignore'):
            x = np.uint32(np_hash_uint(x))
            y = np.uint32(np_hash_uint(y))
            r = np_hash_uint(x * np.uint32(0x85EBCA6B) ^ y * np.uint32(0xC2B2AE35))
        return r / np.float32(0xFFFFFFFF) if normalize else x

    @classmethod
    def pt_noise2(cls, p: torch.Tensor) -> torch.Tensor:
        """
        2D value noise scramble.

        :param p: Input tensor of shape [2, H, W].
        :return: Scrambled tensor of shape [2, H, W], float32 in [0, 1].
        """

        c1 = torch.tensor([127.1, 311.7], dtype=p.dtype, device=p.device).view(2,1,1)
        c2 = torch.tensor([269.5, 183.3], dtype=p.dtype, device=p.device).view(2,1,1)

        dx = torch.sum(p * c1, dim=0)
        dy = torch.sum(p * c2, dim=0)
        r  = torch.stack([dx, dy], dim=0)

        return (torch.sin(r) * 18.5453).remainder(1.0)

    @classmethod
    def pt_noise3(cls, p: torch.Tensor) -> torch.Tensor:
        """
        3D value noise scramble.

        :param p: Input tensor of shape [3, H, W].
        :return: Scrambled tensor of shape [3, H, W], float32 in [0, 1].
        """
        c1 = torch.tensor([127.1, 311.7, 419.2], dtype=p.dtype, device=p.device).view(3,1,1)
        c2 = torch.tensor([269.5, 183.3, 371.9], dtype=p.dtype, device=p.device).view(3,1,1)
        c3 = torch.tensor([419.2, 371.9, 629.1], dtype=p.dtype, device=p.device).view(3,1,1)

        dx = torch.sum(p * c1, dim=0)
        dy = torch.sum(p * c2, dim=0)
        dz = torch.sum(p * c3, dim=0)
        r  = torch.stack([dx, dy, dz], dim=0)

        return (torch.sin(r) * 43758.5453).remainder(1.0)