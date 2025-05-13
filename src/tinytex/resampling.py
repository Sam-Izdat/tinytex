import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

from .util import *

class Resampling:

    """Image resizing and padding."""

    err_size = "tensor must be sized [C, H, W] or [N, C, H, W]"

    @classmethod
    def tile(cls, im:torch.Tensor, shape:tuple) -> torch.Tensor:
        """
        Tile/repeat image tensor to match target shape.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W].
        :param shape: Target shape as (height, width) tuple.
        :return: Padded image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        new_H, new_W = shape[0], shape[1]
        H, W = im.shape[2:]
        if H == new_H and W == new_W: return im.squeeze(0) if nobatch else im
        assert H <= new_H and W <= new_W, "target shape cannot be smaller than input shape"
        h_tiles = (new_H // H) + 1
        w_tiles = (new_W // W) + 1
        tiled_tensor = im.repeat(1, 1, h_tiles, w_tiles)
        tiled_tensor = tiled_tensor[..., :new_H, :new_W]
        return tiled_tensor.squeeze(0) if nobatch else tiled_tensor

    @classmethod
    def tile_n(cls, im:torch.Tensor, repeat_h:int, repeat_w:int):
        """
        Tile/repeat image tensor by number of repetitions.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W]
        :param repeat_h: Number of times to repeat image vertically.
        :param repeat_w: Number of times to repeat image horizontally.
        :return: Padded image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)        
        H, W = im.shape[2:]
        tiled_tensor = cls.tile(im, shape=(H*repeat_h, W*repeat_w))
        return tiled_tensor.squeeze(0) if nobatch else tiled_tensor

    @classmethod
    def tile_to_square(cls, im:torch.Tensor, target_size:int) -> torch.Tensor:
        """
        Tile image tensor to square dimensions of target size.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W].
        :return: Padded image tensor sized [C, H, W] or [N, C, H, W] where H = W.
        """
        # Uses numpy for legacy reasons, but can be reworked to use torch tile method above.
        # F.pad won't tile the image to arbitrary size.
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        N, C, H, W = im.size()
        if H == new_H and W == new_W: return im.squeeze(0) if nobatch else im
        assert H <= target_size and W <= target_size, "target shape cannot be smaller than input shape"
        np_image = im.permute(0, 2, 3, 1).numpy()
        tiled_image = np.ndarray([N, target_size, target_size, C])
        for i in range(im.shape[0]):
            h_tiles = int(np.ceil(target_size / np_image[i].shape[1]))
            v_tiles = int(np.ceil(target_size / np_image[i].shape[0]))
            tiled_image[i] = np.tile(np_image[i], (v_tiles, h_tiles, 1))[:target_size, :target_size, :]
        res = torch.from_numpy(tiled_image).permute(0, 3, 1, 2)
        return res.squeeze(0) if nobatch else res

    @classmethod
    def crop(cls, im:torch.Tensor, shape:tuple, start:tuple=(0, 0)):
        """
        Crop image tensor to maximum target shape, if and only if a crop box target dimension 
        is smaller than the boxed image dimension. Returned tensor can be smaller than target 
        shape, depending on input image shape - i.e. no automatic padding.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W].
        :param shape: Target shape as (height, width) tuple.
        :param start: Top-left corner coordinates of the crop box as (top, left) tuple.
        :return: Cropped image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        assert start[0] < im.size(2) and start[1] < im.size(2), \
            "crop box start dimensions must be smaller than image dimensions"
        H, W = min(im.size(2) - start[0], shape[0]), min(im.size(3) - start[1], shape[1])
        if H == im.size(2) and W == im.size(3): return im.squeeze(0) if nobatch else im
        res = TF.crop(im, top=start[0], left=start[1], height=H, width=W)
        return res.squeeze(0) if nobatch else res

    @classmethod
    def resize(cls, im:torch.Tensor, shape:tuple, mode:str='bilinear', iterative_downsample=False):    
        """
        Resize image tensor to target shape.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W].
        :param shape: Target shape as (height, width) tuple.
        :param mode: Resampleing algorithm ('nearest' | 'linear' | 'bilinear' | 'bicubic' | 'area').
        :param iterative_downsample: Iteratively average pixels if image dimension must be reduced 2x or more.
        :return: Resampled image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        res = None

        if iterative_downsample:
            # Use symmetric pooling until both dimensions are < 2x target.
            while im.size(2) >= shape[1] * 2 and im.size(3) >= shape[0] * 2:
                im = F.avg_pool2d(im, kernel_size=2, stride=2)

        res = F.interpolate(im, size=shape, mode=mode, align_corners=False)
        return res.squeeze(0) if nobatch else res

    @classmethod
    def resize_se(cls, im:torch.Tensor, size:int, mode:str='bilinear', iterative_downsample=False) -> torch.Tensor:
        """
        Resize image tensor by shortest edge, constraining proportions.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W]
        :param size: Target size for shortest edge
        :param mode: Resampleing algorithm ('nearest' | 'linear' | 'bilinear' | 'bicubic' | 'area')
        :param iterative_downsample: Iteratively average pixels if image dimension must be reduced 2x or more.
        :return: Resampled image tensor sized [C, H, W] or [N, C, H, W]
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        H, W = im.shape[2:]
        if min(H, W) == size: return im.squeeze(0) if nobatch else im
        scale = size / min(H, W)
        new_h = int(np.ceil(H * scale))
        new_w = int(np.ceil(W * scale))

        if iterative_downsample:
            # Use symmetric pooling until both dimensions are < 2x target.
            while im.size(2) >= new_h * 2 and im.size(3) >= new_w * 2:
                im = F.avg_pool2d(im, kernel_size=2, stride=2)

        
        res = F.interpolate(im, size=(new_h, new_w), mode=mode, align_corners=False)
        return res.squeeze(0) if nobatch else res

    @classmethod
    def resize_le(cls, im:torch.Tensor, size:int, mode:str='bilinear', iterative_downsample=False) -> torch.Tensor:
        """
        Resize image tensor by longest edge, constraining proportions.

        :param im: Image tensor sized [C, H, W] or [N, C, H, W]
        :param size: Target size for longest edge
        :param mode: Resampleing algorithm ('nearest' | 'linear' | 'bilinear' | 'bicubic' | 'area')
        :param iterative_downsample: Iteratively average pixels if image dimension must be reduced 2x or more.
        :return: Resampled image tensor sized [C, H, W] or [N, C, H, W]
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        H, W = im.shape[2:]
        if max(H, W) == size: return im.squeeze(0) if nobatch else im
        scale = size / max(H, W)
        new_h = int(np.ceil(H * scale))
        new_w = int(np.ceil(W * scale))

        if iterative_downsample:
            # Use symmetric pooling until both dimensions are < 2x target.
            while im.size(2) >= new_h * 2 and im.size(3) >= new_w * 2:
                im = F.avg_pool2d(im, kernel_size=2, stride=2)

        
        res = F.interpolate(im, size=(new_h, new_w), mode=mode, align_corners=False)
        return res.squeeze(0) if nobatch else res

    def resize_le_to_next_pot(im:torch.Tensor, mode:str='bilinear'):
        """
        Resize image tensor by longest edge up to next higher power-of-two, constraining proportions.

        :param torch.tensor im: Image tensor sized [C, H, W] or [N, C, H, W].
        :return: Resampled image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        H, W = im.shape[2:]
        max_dim = max(H, W)
        size = next_pot(max_dim)
        res = cls.resize_le(im, size=size, mode=mode)
        return res.squeeze(0) if nobatch else res

    @classmethod
    def pad_rb(cls, im:torch.Tensor, shape:tuple, mode:str='replicate') -> torch.Tensor:
        """
        Pad image tensor to the right and bottom to target shape.

        :param torch.Tensor im: Image tensor sized [C, H, W] or [N, C, H, W].
        :param mode: Padding algorithm ('constant' | 'reflect' | 'replicate' | 'circular').
        :return: Padded image tensor sized [C, H, W] or [N, C, H, W].
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)

        H, W = im.shape[2:]
        th, tw = shape[0], shape[1]
        assert tw > W and th > H, "target dimensions must be larger than image dimensions"

        pad_h = int(th - H)
        pad_w = int(tw - W)
        padding = (0, pad_w, 0, pad_h)
        if (H < th / 2. or W < tw / 2.) and (mode == "circular" or mode == "reflect"): 
            if mode == "reflect": raise Exception("target size too big for reflect mode")
            # Can't use torch pad, because dimensions won't allow it - fall back to manual repeat.
            padded_image = cls.tile(im.cpu(), shape=shape).to(im.device)
        else:
            padded_image = F.pad(im, padding, mode=mode, value=0)

        if nobatch: padded_image = padded_image.squeeze(0)
        return padded_image

    @classmethod
    def pad_to_next_pot(cls, im:torch.Tensor, mode:str='replicate') -> torch.Tensor:
        """
        Pad image tensor to next highest power-of-two square dimensions.

        :param torch.Tensor im: image tensor sized [C, H, W] or [N, C, H, W]
        :param mode: padding algorithm ('constant' | 'reflect' | 'replicate' | 'circular')
        :return: padded image tensor sized [C, H, W] or [N, C, H, W] where H = W
        """
        ndim = len(im.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: im = im.unsqueeze(0)
        H, W = im.shape[2:]
        size = next_pot(max(H, W))
        padded_image = cls.pad_rb(im, shape=(size, size), mode=mode)
        if nobatch: padded_image = padded_image.squeeze(0)
        return padded_image