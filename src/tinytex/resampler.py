import torch
import torch.nn.functional as F
import numpy as np

from util import *

class Resampler:
    @classmethod
    def tile(cls, im:torch.Tensor, target_shape:tuple) -> torch.Tensor:
        """
        Tile image tensor to target shape.

        :param torch.Tensor im: image tensor sized [C, H, W] or [N, C, H, W]
        :param tuple target_shape: target shape as (height, width) tuple
        :return: padded image tensor sized [C, H, W] or [N, C, H, W] where H = W
        :rtype: torch.tensor 
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, "tensor must be sized [C, H, W] or [N, C, H, W]"
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        new_H, new_W = target_shape[0], target_shape[1]
        H, W = im.shape[2:]
        assert H < new_H and W < new_W, "target shape must be larger than input shape"
        h_tiles = (new_H // H) + 1
        w_tiles = (new_W // W) + 1
        tiled_tensor = im.repeat(1, 1, h_tiles, w_tiles)
        tiled_tensor = tiled_tensor[..., :new_H, :new_W]
        return tiled_tensor.squeeze(0) if nobatch else tiled_tensor

    @classmethod
    def tile_to_square(cls, im:torch.Tensor, target_size:int) -> torch.Tensor:
        """
        Tile image tensor to square dimensions of target size.

        :param torch.Tensor im: image tensor sized [C, H, W] or [N, C, H, W]
        :return: padded image tensor sized [C, H, W] or [N, C, H, W] where H = W
        ("_pt" because i/o is tensors, but uses numpy as torch complains if padding extends too far)
        """
        # Uses numpy for legacy reasons, but can be reworked to use torch tile method above.
        # F.pad won't tile the image to arbitrary size.
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, "tensor must be sized [C, H, W] or [N, C, H, W]"
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        N, C, H, W = im.size()
        assert H < target_size and W < target_size, "target shape must be larger than input shape"
        np_image = im.permute(0, 2, 3, 1).numpy()
        tiled_image = np.ndarray([N, target_size, target_size, C])
        for i in range(im.shape[0]):
            h_tiles = int(np.ceil(target_size / np_image[i].shape[1]))
            v_tiles = int(np.ceil(target_size / np_image[i].shape[0]))
            tiled_image[i] = np.tile(np_image[i], (v_tiles, h_tiles, 1))[:target_size, :target_size, :]
            
        res = torch.from_numpy(tiled_image).permute(0, 3, 1, 2)
        return res.squeeze(0) if nobatch else res

    @classmethod
    def resize(cls, im:torch.Tensor, target_shape:tuple, mode:str='bilinear'):    
        """
        Resize image tensor longest to target shape.

        :param torch.Tensor im: image tensor sized [C, H, W] or [N, C, H, W]
        :param tuple target_shape: target shape as (height, width) tuple
        :param str mode: resampleing algorithm ('nearest' | 'linear' | 'bilinear' | 'bicubic')
        :return: resampled image tensor sized [C, H, W] or [N, C, H, W]
        :rtype: torch.tensor 
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, "tensor must be sized [C, H, W] or [N, C, H, W]"
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        res = F.interpolate(im, size=target_shape, mode=mode, align_corners=False)
        return res.squeeze(0) if nobatch else res

    def resize_to_next_pot(im:torch.Tensor):
        """
        Resize image longest edge to next highest power-of-two, constraining proportions

        :param torch.tensor im: image tensor sized [N, C, H, W]
        :return: resampled image tensor sized [N, C, H, W]
        :rtype: torch.tensor 
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, "tensor must be sized [C, H, W] or [N, C, H, W]"
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        H, W = im.shape[2:]
        max_dim = max(H, W)
        s = next_pot(max_dim)
        res = cls.resize_longest_edge(im, s)
        return res.squeeze(0) if nobatch else res

    # TODO: pad
    #............
    #............

    @classmethod
    def pad_to_next_pot(cls, im:torch.Tensor, mode:str='replicate', allow_lower:bool=False) -> torch.Tensor:
        """
        Pad image tensor to next highest power-of-two square dimensions.

        :param torch.Tensor im: image tensor sized [C, H, W] or [N, C, H, W]
        :return: padded image tensor sized [C, H, W] or [N, C, H, W] where H = W
        :rtype: torch.tensor 
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, "tensor must be sized [C, H, W] or [N, C, H, W]"
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        H, W = im.shape[2:]

        # Compute the nearest power-of-two size
        size = 2 ** int(np.ceil(np.log2(max(H, W))))
        if not allow_lower and size < max(W, H):
            size *= 2

        # Compute the padding needed to make the image a square or power-of-two
        pad_h = size - H
        pad_w = size - W
        padding = (0, pad_w, 0, pad_h)

        if min(H, W) < (size / 2) and (mode == "circular" or mode == "reflect"): 
            if mode == "reflect": raise Exception("target size too big for reflect mode")
            # Can't use torch pad, because dimensions won't allow it - fall back to manual repeat.
            padded_image = cls.tile_to_square(im.cpu(), size=size).to(im.device)
        else:
            # Pad the image to the nearest power-of-two square
            padded_image = F.pad(im, padding, mode=mode, value=0)

        if nobatch: padded_image = padded_image.squeeze(0)
        return padded_image

    @classmethod
    def resize_longest_edge(cls, im:torch.Tensor, target_size:int, mode:str='bilinear') -> torch.Tensor:
        """
        Resize image tensor longest edge to arbitrary size, constraining proportions.

        :param torch.Tensor im: image tensor sized [C, H, W] or [N, C, H, W]
        :param int target_size: target size for longest edge
        :param str mode: resampleing algorithm ('nearest' | 'linear' | 'bilinear' | 'bicubic')
        :return: resampled image tensor sized [C, H, W] or [N, C, H, W]
        :rtype: torch.tensor 
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, "tensor must be sized [C, H, W] or [N, C, H, W]"
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        H, W = im.shape[2:]
        if max(H, W) == target_size: return im
        scale = target_size / max(H, W)
        new_h = math.ceil(H * scale)
        new_w = math.ceil(W * scale)
        res = F.interpolate(im, size=(new_h, new_w), mode=mode, align_corners=False)
        return res.squeeze(0) if nobatch else res