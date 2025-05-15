import random
import torch
import numpy as np

import typing
from typing import Union

from tinycio import fsio

from .util import *
from .resampling import Resampling

class Atlas:

    """Texture atlas packing and sampling."""

    min_size = 64
    max_size = 8192
    force_square = False

    block_size = 256

    def __init__(self, min_size=64, max_size=8192, force_square=False):
        self.min_size = min_size
        self.max_size = max_size
        self.force_square = force_square
        self._textures = {}  # key: name, value: tensor
        self.atlas = None
        self.index = None
    
    def add(self, key: str, tensor: torch.Tensor) -> None:
        """
        Add a named texture to the atlas.

        :param key: Identifier for the texture.
        :param tensor: Image tensor in CHW format.
        """
        self._textures[key] = tensor
    
    class _TextureRect:
        def __init__(self, tensor, idx, key):
            self.tensor = tensor
            self.idx = idx
            self.key = key
            self.x = 0
            self.y = 0
            self.was_packed = False

    err_out_of_bounds = 'failed to to fit textures into atlas'

    def pack(self,
             max_h: int = 0,
             max_w: int = 0,
             crop: bool = True,
             row: bool = False,
             sort: str = 'height') -> tuple:
        """
        Pack added textures into a single atlas image.

        :param max_h: Max atlas height. If 0, auto-expand.
        :param max_w: Max atlas width. If 0, auto-expand.
        :param crop: Crop output to tight bounding box.
        :param row: Use row-based packing instead of rectangle packing.
        :param sort: Sorting mode for texture order ('height', 'width', 'area').

        :return: (Atlas tensor, index dictionary with coordinates per texture).
        """
        H, W = max_h, max_w
        auto_crop = crop and not self.force_square
        if W == 0 or H == 0:
            i = 0
            auto_width, auto_height = W, H
            while auto_height < self.max_size and auto_width < self.max_size:
                if self.force_square:
                    auto_height = (H or int(next_pot(self.min_size)) << i)
                    auto_width = (W or int(next_pot(self.min_size)) << i)
                else:
                    if H == 0 and (W != 0 or auto_width > auto_height):
                        auto_height = (H or int(next_pot(self.min_size)) << i)
                    elif W == 0 and (H != 0 or auto_height >= auto_width):
                        auto_width = (W or int(next_pot(self.min_size)) << i)
                    else:
                        raise Exception('undefined') # should not happen
                atlas, index = self.__row_pack(self._textures, (auto_height, auto_width), sort=sort, auto_crop=auto_crop) if row \
                    else self.__rect_pack(self._textures, (auto_height, auto_width), sort=sort, auto_crop=auto_crop)
                if not atlas is False:
                    self.atlas = atlas
                    self.index = index
                    return self.atlas, self.index
                i += 1
            raise Exception(self.err_out_of_bounds + f" at w {auto_width} h {auto_height}")

        res = self.__row_pack(textures, (H, W), auto_crop=auto_crop, sort=sort, must_succeed=True) if row \
            else self.__rect_pack(textures, (H, W), auto_crop=auto_crop, sort=sort, must_succeed=True) 

        self.atlas = res.atlas
        self.index = res.index
        return self.atlas, self.index

    @classmethod
    def from_dir(cls,
                 path: str,
                 ext: str = '.png',
                 channels: int = 3,
                 allow_mismatch: bool = True,
                 max_h: int = 0,
                 max_w: int = 0,
                 crop: bool = True,
                 row: bool = False,
                 sort: str = 'height') -> 'Atlas':
        """
        Load all matching image files from a directory and pack them.

        :param path: Path to image directory.
        :param ext: File extension to match.
        :param channels: Expected channel count.
        :param allow_mismatch: Auto pad/trim to match channels.
        :param max_h: Optional max height for atlas.
        :param max_w: Optional max width for atlas.
        :param crop: Whether to crop excess space after packing.
        :param row: Whether to use row packing.
        :param sort: Sorting mode.

        :return: Packed Atlas instance.
        """
        atlas = cls()
        for fn in os.listdir(path):
            if not fn.endswith(ext): continue
            im = fsio.load_image(os.path.join(path, fn))
            if allow_mismatch:
                if im.size(0) < channels: im = im.repeat(channels, 1, 1)
                if im.size(0) > channels: im = im[:channels]
            else:
                assert im.size(0) == channels, f"channel mismatch: {fn}"
            key = os.path.splitext(fn)[0]
            atlas.add(key, im)
        assert len(atlas._textures) > 0, "no images found"
        atlas.pack(max_h=max_h, max_w=max_w, crop=crop, row=row, sort=sort)
        return atlas

    def sample(self, key: Union[str, int]) -> torch.Tensor:
        """
        Retrieve a single texture by name or index.

        :param key: Texture name or integer index.
        :return: Image tensor sliced from the atlas.
        """
        assert len(self.index) > 0, "index is empty"
        if isinstance(key, str) and key in self.index:
            x0, y0, x1, y1 = self.index[key]
            return self.atlas[:, int(y0):int(y1), int(x0):int(x1)]
        elif isinstance(key, int):
            x0, y0, x1, y1 = list(self.index.values())[key]
            return self.atlas[:, int(y0):int(y1), int(x0):int(x1)]
        else:
            raise KeyError("key not found in index")

    def sample_random(self) -> torch.Tensor:
        """
        Retrieve a randomly chosen texture from the atlas.

        :return: Image tensor sliced from the atlas.
        """
        assert len(self.index) > 0, "index is empty"
        x0, y0, x1, y1 = random.choice(list(self.index.values()))
        return self.atlas[:, int(y0):int(y1), int(x0):int(x1)]


    @classmethod
    def __sp_push_back(cls, spaces, space):
        return space if spaces == None else torch.cat([spaces, space], dim=0)

    @classmethod
    def __sp_rem(cls, spaces, idx):
        return torch.cat((spaces[:idx], spaces[idx+1:]))

    # https://github.com/TeamHypersomnia/rectpack2D?tab=readme-ov-file#algorithm
    # A bit slower. Suitable for high variance.
    @classmethod
    def __rect_pack(cls, 
        textures:dict, 
        shape:tuple, 
        auto_crop:bool=True, 
        sort:str='height', 
        must_succeed:bool=False) -> (torch.Tensor, tuple):
        texture_rects = []
        max_w, max_h = 0, 0
        for k, v in enumerate(textures):
            texture_rects.append(cls._TextureRect(textures[v], idx=k, key=v))

        atlas_height = shape[0]
        atlas_width = shape[1]

        atlas = torch.zeros(texture_rects[0].tensor.size(0), atlas_height, atlas_width)

        # x0, y0, x1, y1
        empty_spaces = None
        empty_spaces = cls.__sp_push_back(empty_spaces, torch.Tensor([[0, 0, atlas_width, atlas_height]]))

        # Sort textures in descending order
        if sort == 'height':
            texture_rects.sort(key=lambda tex: tex.tensor.size(1), reverse=True)
        elif sort == 'width':
            texture_rects.sort(key=lambda tex: tex.tensor.size(2), reverse=True)
        elif sort == 'area':
            texture_rects.sort(key=lambda tex: (tex.tensor.size(1) * tex.tensor.size(2)), reverse=True)
        else:
            raise Exception(f'unrecognized sort order: {sort}')

        for i, tex in enumerate(texture_rects):
            tex_h, tex_w = tex.tensor.shape[1:]
            best_fit_area = None
            best_fit_idx = None
            for space_idx in range(empty_spaces.size(0)):
                space_idx = empty_spaces.size(0) - 1 - space_idx
                space = empty_spaces[space_idx:space_idx+1,...]
                sp_w, sp_h = space[0,2].item(), space[0,3].item()
                if sp_w >= tex_w and sp_h >= tex_h:
                    if best_fit_area == None or best_fit_area > sp_w * sp_h:
                        best_fit_area = sp_w * sp_h
                        best_fit_idx = space_idx

            if best_fit_idx == None:
                if must_succeed:
                    raise Exception(cls.err_out_of_bounds + f" at w {atlas_width} h {atlas_height}")
                else:
                    return False, False

            space = empty_spaces[best_fit_idx:best_fit_idx+1,...]
            sp_x, sp_y = space[0,0].item(), space[0,1].item()
            sp_w, sp_h = space[0,2].item(), space[0,3].item()
            atlas[...,
                int(sp_y):int(sp_y+tex_h), 
                int(sp_x):int(sp_x+tex_w)] = tex.tensor
            tex.x = sp_x
            tex.y = sp_y
            tex.was_packed = True
            if sp_w > tex_w and sp_h > tex_h:
                split1 = torch.Tensor([[
                    sp_x,
                    sp_y+tex_h,
                    sp_w,
                    sp_h-tex_h]])
                split2 = torch.Tensor([[
                    sp_x+tex_w,
                    sp_y,
                    sp_w-tex_w,
                    tex_h]])
                empty_spaces = cls.__sp_rem(empty_spaces, best_fit_idx)
                if split2[0,2].item()*split2[0,3].item() > split1[0,2].item()*split1[0,3].item():
                    empty_spaces = cls.__sp_push_back(empty_spaces, split2)
                    empty_spaces = cls.__sp_push_back(empty_spaces, split1)
                else:
                    empty_spaces = cls.__sp_push_back(empty_spaces, split1)
                    empty_spaces = cls.__sp_push_back(empty_spaces, split2)
            elif sp_w > tex_w: 
                split = torch.Tensor([[
                    sp_x+tex_w,
                    sp_y,
                    sp_w-tex_w,
                    tex_h]])
                empty_spaces = cls.__sp_rem(empty_spaces, best_fit_idx)
                empty_spaces = cls.__sp_push_back(empty_spaces, split)
            elif sp_h > tex_h:
                split = torch.Tensor([[
                    sp_x,
                    sp_y+tex_h,
                    sp_w,
                    sp_h-tex_h]])
                empty_spaces = cls.__sp_rem(empty_spaces, best_fit_idx)
                empty_spaces = cls.__sp_push_back(empty_spaces, split)
            elif sp_h == tex_h and sp_w == tex_w:
                empty_spaces = cls.__sp_rem(empty_spaces, best_fit_idx)
            else:
                raise Exception(cls.err_out_of_bounds, + f" at w {atlas_width} h {atlas_height}")

        # Sort textures by input order
        texture_rects.sort(key=lambda tex: tex.idx)
        index = {}
        for tex in texture_rects: 
            index[tex.key] = (tex.x, tex.y, tex.x+tex.tensor.size(2), tex.y+tex.tensor.size(1))
            if (tex.x+tex.tensor.size(2)) > max_w: max_w = tex.x+tex.tensor.size(2)
            if (tex.y+tex.tensor.size(1)) > max_h: max_h = tex.y+tex.tensor.size(1)

        if auto_crop: atlas = Resampling.crop(atlas, (int(max_h), int(max_w)))
        return atlas, index                        

    # https://www.david-colson.com/2020/03/10/exploring-rect-packing.html
    # Faster. Suitable for low variance.
    @classmethod
    def __row_pack(cls, 
        textures:dict, 
        shape:tuple, 
        auto_crop:bool=True, 
        sort:str='height', 
        must_succeed:bool=False) -> (torch.Tensor, tuple):
        texture_rects = []
        max_w, max_h = 0, 0
        for k, v in enumerate(textures):
            texture_rects.append(cls._TextureRect(textures[v], idx=k, key=v))

        # Sort textures in descending order
        if sort == 'height':
            texture_rects.sort(key=lambda tex: tex.tensor.size(1), reverse=True)
        elif sort == 'width':
            texture_rects.sort(key=lambda tex: tex.tensor.size(2), reverse=True)
        elif sort == 'area':
            texture_rects.sort(key=lambda tex: (tex.tensor.size(1) * tex.tensor.size(2)), reverse=True)
        else:
            raise Exception(f'unrecognized sort order: {sort}')

        atlas_height = shape[0]
        atlas_width = shape[1]

        atlas = torch.zeros(texture_rects[0].tensor.size(0), atlas_height, atlas_width)

        x_pos = 0
        y_pos = 0
        largest_height_this_row = 0

        # Loop over all the textures
        for tex in texture_rects:
            tex_h, tex_w = tex.tensor.shape[1:]
            # If this texture will go past the width of the atlas,
            # loop around to the next row, using the largest height from the previous row
            if (x_pos + tex.tensor.size(2)) > atlas_width:
                y_pos = y_pos + largest_height_this_row
                x_pos = 0
                largest_height_this_row = 0

            if (y_pos + tex_h) > atlas_height or (x_pos + tex_w) > atlas_width:
                if must_succeed:
                    raise Exception(cls.err_out_of_bounds, + f" at w {atlas_width} h {atlas_height}")
                else:
                    return False, False

            tex.x = x_pos
            tex.y = y_pos

            atlas[:, y_pos:y_pos + tex_h, x_pos:x_pos + tex_w] = tex.tensor

            x_pos += tex_w

            # Save largest height in the new row
            if tex_h > largest_height_this_row:
                largest_height_this_row = tex_h

            tex.was_packed = True

        # Sort textures by input order
        index = {}
        for tex in texture_rects: 
            index[tex.key] = (tex.x, tex.y, tex.x+tex.tensor.size(2), tex.y+tex.tensor.size(1))
            if (tex.x+tex.tensor.size(2)) > max_w: max_w = tex.x+tex.tensor.size(2)
            if (tex.y+tex.tensor.size(1)) > max_h: max_h = tex.y+tex.tensor.size(1)

        if auto_crop: atlas = Resampling.crop(atlas, (max_h, max_w))
        return atlas, index

    def generate_mask(self, shape: tuple, scale: float = 1.0, samples: int = 2) -> torch.Tensor:
        """
        Generate a tiling mask by randomly overlaying textures.

        :param shape: Output (H, W) of the canvas.
        :param scale: Relative size of overlays.
        :param samples: Density multiplier for overlays.
        :return: Output image tensor.
        """

        output_size = (self.atlas.size(0), shape[0], shape[1])
        output_image = torch.zeros(output_size)

        num_overlays = int(((shape[0] * shape[1]) / scale / self.block_size**2) * samples)

        for _ in range(num_overlays):

            # Load the overlay texture
            overlay_texture = self.sample_random()
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