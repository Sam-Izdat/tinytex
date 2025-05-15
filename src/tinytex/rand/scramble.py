from __future__ import annotations
import typing
from typing import Union
import torch
import numpy as np

def pt_noise2(p: torch.Tensor) -> torch.Tensor:
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

def pt_noise3(p: torch.Tensor) -> torch.Tensor:
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