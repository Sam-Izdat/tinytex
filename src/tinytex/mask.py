import torch
import os
import random

from tinycio import fsio
from tinytex import Resampling

from .atlas import Atlas

class MaskGenerator:

    def __init__(self, atlas, index):
        self.atlas = atlas
        self.index = index

    def generate(self, shape, scale=1.0, samples=2):
        """
        Generate tiling mask by randomly sampling the texture atlas and randomly positioning textures on the canvas.
        """
        block_size = 256

        output_size = (self.atlas.size(0), shape[0], shape[1])
        output_image = torch.zeros(output_size)

        num_overlays = int(((shape[0] * shape[1]) / scale / block_size**2) * samples)

        for _ in range(num_overlays):

            # Load the overlay texture
            overlay_texture = Atlas.sample_random(self.atlas, self.index)
            overlay_size = overlay_texture.size()[1:]
            overlay_texture = Resampling.resize_le(overlay_texture, max(overlay_size[0], overlay_size[1]) * scale)
            overlay_size = overlay_texture.size()[1:]

            # Random position from the top-left
            position = (random.randint(0, shape[0] - 1), random.randint(0, shape[1] - 1))
            wrap_width, wrap_height = 0, 0

            # If overlay exceeds canvas borders, draw truncated part on the opposite side
            if position[0] + overlay_size[0] > shape[0]:
                wrap_height = (position[0] + overlay_size[0]) % shape[0]

            if position[1] + overlay_size[1] > shape[1]:
                wrap_width = (position[1] + overlay_size[1]) % shape[1]

            if wrap_width == 0 and wrap_height == 0:
                output_image[:, position[0]:position[0]+overlay_size[0], position[1]:position[1]+overlay_size[1]] += overlay_texture
            else:
                max_pos_h = position[0]+overlay_size[0]-wrap_height
                max_pos_w = position[1]+overlay_size[1]-wrap_width

                # non-overflow top-left quadrant
                output_image[:, position[0]:max_pos_h, position[1]:max_pos_w] += \
                    overlay_texture[:, 0:overlay_size[0]-wrap_height, 0:overlay_size[1]-wrap_width]

                # overflow top-right quadrant
                if wrap_width > 0:
                    output_image[:, position[0]:max_pos_h, 0:wrap_width] += \
                        overlay_texture[:, 0:overlay_size[0]-wrap_height, overlay_size[1]-wrap_width:overlay_size[1]]

                # overflow bottom-left quadrant
                if wrap_height > 0:
                    output_image[:, 0:wrap_height, position[1]:max_pos_w] += \
                        overlay_texture[:, overlay_size[0]-wrap_height:overlay_size[0], 0:overlay_size[1]-wrap_width]

                # overflow bottom-right quadrant
                if wrap_height > 0 and wrap_height > 0:
                    output_image[:, 0:wrap_height, 0:wrap_width] += \
                        overlay_texture[:, overlay_size[0]-wrap_height:overlay_size[0], overlay_size[1]-wrap_width:overlay_size[1]]

        return output_image