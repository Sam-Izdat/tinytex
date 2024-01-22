import torch
import numpy as np

from combomethod import combomethod

from .util import *

class Atlas:

    """Texture atlas packing."""

    min_auto_size = 64
    max_auto_size = 8192
    force_auto_square = False

    def __init__(self, min_auto_size=64, max_auto_size=8192, force_auto_square=False):
        self.min_auto_size = min_auto_size
        self.max_auto_size = max_auto_size
        self.force_auto_square = force_auto_square

    class _TextureRect:
        def __init__(self, tensor, idx):
            self.tensor = tensor
            self.idx = idx
            self.x = 0
            self.y = 0
            self.was_packed = False

    err_out_of_bounds = 'failed to to fit textures into atlas'

    @combomethod
    def pack(cls, textures:list, width:int=0, height:int=0, row_pack:bool=False) -> (torch.Tensor, list):
        if width == 0 or height == 0:
            i = 0
            auto_width, auto_height = width, height
            while auto_height < cls.max_auto_size and auto_width < cls.max_auto_size:
                if cls.force_auto_square:
                    auto_height = (height or int(next_pot(cls.min_auto_size)) << i)
                    auto_width = (width or int(next_pot(cls.min_auto_size)) << i)
                else:
                    if height == 0 and (width != 0 or auto_width > auto_height):
                        auto_height = (height or int(next_pot(cls.min_auto_size)) << i)
                    elif width == 0 and (height != 0 or auto_height >= auto_width):
                        auto_width = (width or int(next_pot(cls.min_auto_size)) << i)
                    else:
                        raise Exception('undefined') # should not happen
                atlas, index = cls.__row_pack(textures, (auto_height, auto_width)) if row_pack \
                    else cls.__rect_pack(textures, (auto_height, auto_width))
                if not atlas is False:
                    return atlas, index
                i += 1
            raise Exception(cls.err_out_of_bounds + f" at w {auto_width} h {auto_height}")
        return cls.__row_pack(textures, (height, width), must_succeed=True) if row_pack \
            else cls.__rect_pack(textures, (height, width), must_succeed=True) 

    @classmethod
    def __sp_push_back(cls, spaces, space):
        return torch.cat([spaces, space], dim=0)

    @classmethod
    def __sp_rem(cls, spaces, idx):
        return torch.cat((spaces[:idx], spaces[idx+1:]))

    # https://github.com/TeamHypersomnia/rectpack2D?tab=readme-ov-file#algorithm
    # A bit slower. Suitable for high variance.
    @classmethod
    def __rect_pack(cls, textures, shape, must_succeed=False):
        texture_rects = []
        for k, v in enumerate(textures):
            texture_rects.append(cls._TextureRect(v, idx=k))

        atlas_height = shape[0]
        atlas_width = shape[1]

        atlas = torch.zeros(texture_rects[0].tensor.size(0), atlas_height, atlas_width)

        # x, y, w, h
        empty_spaces = torch.Tensor([[0,0,0,0]])
        empty_spaces = cls.__sp_push_back(empty_spaces, torch.Tensor([[0, 0, atlas_width, atlas_height]]))

        # Sort textures by height in descending order
        texture_rects.sort(key=lambda tex: tex.tensor.size(1), reverse=True)

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
        index = [(tex.x, tex.y, tex.x+tex.tensor.size(2), tex.y+tex.tensor.size(1)) for tex in texture_rects]
        return atlas, index                        

    # https://www.david-colson.com/2020/03/10/exploring-rect-packing.html
    # Faster. Suitable for low variance.
    @classmethod
    def __row_pack(cls, textures, shape, must_succeed=False):
        texture_rects = []
        for k, v in enumerate(textures):
            texture_rects.append(cls._TextureRect(v, idx=k))

        # Sort textures by height in descending order
        texture_rects.sort(key=lambda tex: tex.tensor.size(1), reverse=True)

        atlas_height = shape[0]
        atlas_width = shape[1]

        atlas = torch.zeros(texture_rects[0].tensor.size(0), atlas_height, atlas_width)

        x_pos = 0
        y_pos = 0
        largest_height_this_row = 0

        # Loop over all the textures
        for tex in texture_rects:
            tex_h, tex_w = tex.tensor.shape[1:]
            # If this texture will go past the width of the atlas
            # Then loop around to the next row, using the largest height from the previous row
            if (x_pos + tex.tensor.size(2)) > atlas_width:
                y_pos = y_pos + largest_height_this_row
                x_pos = 0
                largest_height_this_row = 0

            # Check for out of bounds
            if (y_pos + tex_h) > atlas_height or (x_pos + tex_w) > atlas_width:
                if must_succeed:
                    raise Exception(cls.err_out_of_bounds, + f" at w {atlas_width} h {atlas_height}")
                else:
                    return False, False
                # break

            # Set position texture
            tex.x = x_pos
            tex.y = y_pos

            # Copy texture to atlas
            atlas[:, y_pos:y_pos + tex_h, x_pos:x_pos + tex_w] = tex.tensor

            # Move to next spot row
            x_pos += tex_w

            # Save largest height in the new row
            if tex_h > largest_height_this_row:
                largest_height_this_row = tex_h

            tex.was_packed = True

        # Sort textures by input order
        texture_rects.sort(key=lambda tex: tex.idx)
        index = [(tex.x, tex.y, tex.x+tex.tensor.size(2), tex.y+tex.tensor.size(1)) for tex in texture_rects]
        return atlas, index