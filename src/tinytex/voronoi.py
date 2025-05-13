from __future__ import annotations
import typing
from typing import Union

import torch

class Voronoi:
    """Voronoi-based generators."""
    @classmethod
    def snap_offset(self, x: torch.Tensor, seed: int):
        """
        Vectorized Voronoi offset and distances.
        Given 2D position `x` and seed, returns F1, F2, best_offset_x, best_offset_y.
        """
        n = torch.floor(x)
        f = fract(x)

        F1 = torch.full_like(f[0], 1e20)  # Initialize with a large value
        F2 = torch.full_like(f[0], 1e20)
        best_offset = torch.zeros_like(f)

        # Vectorize the loop by using broadcasting
        offsets = []
        for j in range(-1, 2):
            for i in range(-1, 2):
                g = torch.tensor([i, j], dtype=torch.float32).to(x.device)
                o = pt_noise2(n + g.view(2,1,1) + fract(seed * G.GR))   # -> [2,H,W]
                r = g.view(2,1,1) - f + o                               # -> [2,H,W]
                d2 = torch.sum(r ** 2, dim=0)

                F1_new = d2 < F1
                F2_new = (d2 < F2) & (~F1_new)

                F2 = torch.where(F1_new, F1, F2)
                F1 = torch.where(F1_new, d2, F1)
                F2 = torch.where(F2_new, d2, F2)

                best_offset = torch.where(F1_new.unsqueeze(0), r, best_offset)

        return torch.stack([torch.sqrt(F1), torch.sqrt(F2), best_offset[0], best_offset[1]])

    @classmethod
    def snap_nearest_as(self, 
        uv_screen: torch.Tensor, 
        vn_scale: float, 
        scale: float, 
        zoom_to: torch.Tensor, 
        aspect: float, 
        seed: int):
        """
        Perturb UVs with a Voronoi field.

        :param uv_screen: UV coordinates of shape [2, H, W] in the range [0, 1]
        :type uv_screen: torch.Tensor
        :param vn_scale: Voronoi scale factor that controls the size of the cells
        :type vn_scale: float
        :param scale: Scene scale (zoom) factor
        :type scale: float
        :param zoom_to: Zoom center as a tensor of shape [2,] in normalized space
        :type zoom_to: torch.Tensor
        :param aspect: Aspect ratio (width/height) of the scene
        :type aspect: float
        :param seed: Random seed used to generate the Voronoi field
        :type seed: int
        :return: A tensor of shape [6, H, W] where:

            - Channel 0: Perturbed UVs (u)
            - Channel 1: Perturbed UVs (v)
            - Channel 2: Voronoi cell ID (cell_id_x)
            - Channel 3: Voronoi cell ID (cell_id_y)
            - Channel 4: Distance from the edge of the Voronoi cell (edge_dist)
            - Channel 5: Distance from the center of the Voronoi cell (center_dist)
            
        :rtype: torch.Tensor
        """
        # Convert input to tensor if needed
        zoom_ten = torch.tensor(zoom_to, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).to(uv_screen.device)
        aspect_ten = torch.tensor([aspect, 1.0]).unsqueeze(-1).unsqueeze(-1).to(uv_screen.device)
        # Compute scene space UV
        uv_scene = zoom_ten + (uv_screen - zoom_ten) * aspect_ten / scale
        xa = uv_scene * vn_scale
        # xa = uv_screen * vn_scale / scale

        # Call the vectorized Voronoi function
        v = self.vn_snap_offset(xa, seed)  # [F1, F2, offset_x, offset_y]
        edge_dist = v[1] - v[0]

        # Snapped scene UV
        snapped_scene = (xa + v[2:4]) / vn_scale
        final_uv = zoom_ten + (snapped_scene - zoom_ten) * scale / aspect_ten

        # Compute cell IDs
        cell_id = torch.floor(xa + v[2:4])

        # Stack the final result
        res = torch.cat([final_uv, cell_id, edge_dist.unsqueeze(0), v[0:1]], dim=0)
        return res