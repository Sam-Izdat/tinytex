import torch
import os
import random

from tinycio import fsio
from tinytex import Resampling

def overlay_generator(atlas, index, shape, scale, samples):
    block_size = 256
    # Get a list of file paths in the folder
    overlay_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

    # Initialize the output image tensor
    output_size = (1, shape[0], shape[1])
    output_image = torch.zeros(output_size)

    # Calculate the number of overlays based on density
    num_overlays = int(((shape[0] * shape[1]) / scale / block_size**2) * samples)

    # Generate random positions, shapes, and rotations for overlays
    for _ in range(num_overlays):
        # Randomly choose a file path
        # overlay_file = random.choice(overlay_files)
        rand_key = random.choice(list(index.keys()))
        overlay_idx = index[rand_key]
        # overlay_file = atlas[random_key]

        # Load the overlay texture
        overlay_texture = fsio.load_image(overlay_file)[3:4,...]
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


    # Ensure values are within the valid range (0, 1)
    output_image = torch.clamp(output_image, 0, 1)

    return output_image

# # Save the resulting image
# fsio.save_image(output_image, save_path)

# # Example usage:
# shape = (512, 512)
# samples = 4.5 # Adjust as needed
# scale = 1.
# folder_path = './spots'
# save_path = './out/res.png'
# overlay_generator(shape, scale, samples, folder_path, save_path)