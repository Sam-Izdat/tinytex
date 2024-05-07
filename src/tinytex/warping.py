import torch
import torch.nn.functional as F
import numpy as np

class Warping:
    
    """Warping and coordinate system translation."""

    @classmethod
    def log_polar(cls, im:torch.Tensor, start_from:int=1, n_angular=None, n_radial=None) -> torch.Tensor:
        """Log polar transform"""
        assert start_from > 0, "must start from 1 or greater"
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        N, C, H, W = im.size()
        H = n_angular or H
        W = n_radial or W
        center = (W // 2, H // 2)

        max_radius = np.sqrt(center[0]**2 + center[1]**2) - 1
        min_radius = start_from

        rho, theta = torch.meshgrid(
            torch.linspace(np.log(min_radius), np.log(max_radius), W),  # Radial coordinates (logarithmic scale)
            torch.linspace(-np.pi, np.pi, H),                           # Angular coordinates
            indexing="xy"
        )
        xx = (center[0] + torch.exp(rho) * torch.cos(theta)) / W * 2. - 1.
        yy = (center[1] + torch.exp(rho) * torch.sin(theta)) / H * 2. - 1.

        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        log_polar_image = F.grid_sample(im, grid, align_corners=True)

        return log_polar_image.squeeze(0) if nobatch else log_polar_image

    @classmethod
    def inverse_log_polar(cls, im:torch.Tensor) -> torch.Tensor:
        """Inverse log polar transform"""
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        N, C, H, W = im.size()
        center = (W // 2, H // 2)
        eps = 1e-7

        max_radius = np.sqrt(center[0]**2 + center[1]**2)
        x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")

        rho = torch.log(torch.sqrt((x - center[0])**2 + (y - center[1])**2 + eps))
        rho[torch.isinf(rho)] = 0.0
        theta = torch.atan2(y - center[1], x - center[0])

        xx = (rho / np.log(max_radius)) * 2. - 1.
        yy = ((theta + np.pi) / (np.pi * 2)) * 2. - 1.

        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        inverse_log_polar_image = F.grid_sample(im, grid, align_corners=True)

        return inverse_log_polar_image.squeeze(0) if nobatch else inverse_log_polar_image