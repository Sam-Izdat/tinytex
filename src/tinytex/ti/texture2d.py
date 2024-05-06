import typing
from typing import Union

import taichi as ti
import taichi.math as tm

import torch
import numpy as np
from enum import IntEnum    

from .prep import *

@ti.data_oriented
class Texture2D:
    """
    Taichi 2D read-write texture. Can be initialized with either texture shape or texture data.

    :param im: Tuple sized (C, H, W) indicating texture shape or image data as [C, H, W] sized 
        PyTorch tensor, NumPy array, Taichi vector or scalar value.
    :type im: tuple | torch.Tensor | numpy.ndarray | float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
    :param generate_mips: Generate mipmaps.
    :param flip_y: Flip y-coordinate before populating.
    """
    def __init__(self, 
        im:Union[tuple, torch.Tensor, np.ndarray, float, tm.vec2, tm.vec3, tm.vec4],
        generate_mips:bool=False,
        flip_y:bool=False
        ):

        self.max_mip = 0
        self.generate_mips = generate_mips
        self.flip_y = flip_y

        FC, FH, FW = 0, 0, 0

        if isinstance(im, tuple):
            assert len(im) == 3, 'tuple must be (C, H, W)'
            assert isinstance(im[0], int) and isinstance(im[1], int) and isinstance(im[0], int), 'tuple values must be int'
            assert im[0] > 0 and im[1] > 0 and im[2] > 0, 'channel and spatial dimensions cannot be zero'
            assert im[0] <= 4, 'image may not have more than 4 channels'
            self.channels = im[0]
            self.height = im[1]
            self.width = im[2]
            FC, FH, FW = self.channels, self.height, (int(self.width * 1.5) if self.generate_mips else self.width)
            if self.generate_mips: self.max_mip = FH.bit_length()
        else:
            self.channels = count_channels(im)

            # prepare data for a ti field
            if self.channels == 1:      im = prep_r(im, self.flip_y)
            elif self.channels == 2:    im = prep_rg(im, self.flip_y)
            elif self.channels == 3:    im = prep_rgb(im, self.flip_y)
            elif self.channels == 4:    im = prep_rgba(im, self.flip_y)
            else: raise Exception(f"Could not populate image data; unexpected number of channels ({self.channels})")

            self.height = im.size(0)
            self.width = im.size(1)
            FC, FH, FW = self.channels, self.height, (int(self.width * 1.5) if self.generate_mips else self.width)
            if self.generate_mips:
                im = im.permute(2, 0, 1)
                C, H, W = im.size()
                tmp = torch.zeros(C, H, W+W//2)
                tmp[:, 0:H, 0:W] = im
                self.max_mip = H.bit_length() - 1
                FC, FH, FW = tmp.size()
                im = tmp.permute(1, 2, 0)

        self.fb = None
        self.fb_snode_tree = None
        self.field = None
        if   self.channels == 1:    self.field = ti.field(dtype=ti.f32)  #ti.Vector.field(1, dtype=ti.f32, shape=(FH, FW))
        elif self.channels == 2:    self.field = ti.field(dtype=tm.vec2) #ti.Vector.field(2, dtype=ti.f32, shape=(FH, FW))
        elif self.channels == 3:    self.field = ti.field(dtype=tm.vec3) #ti.Vector.field(3, dtype=ti.f32, shape=(FH, FW))
        elif self.channels == 4:    self.field = ti.field(dtype=tm.vec4) #ti.Vector.field(4, dtype=ti.f32, shape=(FH, FW))
        self.fb = ti.FieldsBuilder()
        self.fb.dense(ti.ij, (FH, FW)).place(self.field)
        self.fb_snode_tree = self.fb.finalize()
        if torch.is_tensor(im): 
            self.__populateprepared(im)

    def destroy(self):
        """
        Destroy texture data and recover allocated memory. 

        .. note::

            This is not done implicitly using :code:`__del__` as that can sometimes cause Taichi to throws errors, 
            for reasons undetermined, as of version 1.7.1.

        """
        
        if self.fb_snode_tree: 
            self.fb_snode_tree.destroy()
            
    def populate(self, im:Union[torch.Tensor, np.ndarray, float, tm.vec2, tm.vec3, tm.vec4]):
        """
        Populate texture with [C, H, W] sized PyTorch tensor, NumPy array, Taichi vector or scalar value.


        :param im: Image data as [C, H, W] sized PyTorch tensor, NumPy array, Taichi vector or scalar value.
        :type im: torch.Tensor | numpy.ndarray | float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
        """
        assert self.channels == count_channels(im), f"image tensor must have {self.channels} channels; got {count_channels(im)}"
        if self.channels == 1:      
            im = prep_r(im, self.flip_y)
        elif self.channels == 2:    
            im = prep_rg(im, self.flip_y)
        elif self.channels == 3:    
            im = prep_rgb(im, self.flip_y)
        elif self.channels == 4:    
            im = prep_rgba(im, self.flip_y)
        else: raise Exception(f"Could not populate image data; unexpected number of channels ({self.channels})")
        height = im.size(0)
        width = im.size(1)
        assert self.width == width and self.height == height, f"expected W {self.width} x H {self.height} image; got W {width} x H {height}"

        if self.generate_mips:
            im = im.permute(2, 0, 1)
            C, H, W = im.size()
            tmp = torch.zeros(C, H, W+W//2)
            tmp[:, 0:H, 0:W] = im
            self.max_mip = H.bit_length() - 1

            # To generate with torch:
            # HO = 0
            # mip = im.clone()
            # for i in range(1, H.bit_length()):
            #     NH, NW = H >> i, max(W >> i, 1)
            #     mip = ttex.Resampling.resize(mip, (NH, NW))
            #     tmp[:,HO:HO+NH,W:W+NW] = mip
            #     HO += NH

            im = tmp.permute(1, 2, 0) 
        self.__populateprepared(im.float()) 

    def __populateprepared(self, im:torch.Tensor):
        if im.size(2) == 1: im = im.squeeze(-1)
        self.field.from_torch(im.float()) 
        if self.generate_mips: self.regenerate_mips()

    def to_tensor(self):
        """Return texture as [C, H, W] sized PyTorch image tensor."""

        return self.field.to_torch().permute(2, 0, 1)[:, 0:self.height, 0:self.width]    

    # def _regenerate_mips_torch(self):
    #     # FIXME: Some kind of bullshit is happening here, and I don't know what it is.
    #     if self.max_mip == 0: return
    #     H, W, C = self.height, self.width, self.channels
    #     HO = 0
    #     tmp = torch.zeros(C, H, W+W//2)
    #     tmp[:, 0:H, 0:W] = self.field.to_torch().permute(2, 0, 1)[:, 0:H, 0:W]
    #     mip = tmp[:, 0:H, 0:W]
    #     self.max_mip = H.bit_length() - 1
    #     for i in range(1, H.bit_length()):
    #         NH, NW = H >> i, max(W >> i, 1)
    #         mip = ttex.Resampling.resize(mip, (NH, NW))
    #         tmp[:,HO:HO+NH,W:W+NW] = mip
    #         HO += NH
    #     self.populate(tmp.clone().permute(1, 2, 0))

    @ti.kernel
    def regenerate_mips(self):
        self._regenerate_mips()

    @ti.func
    def _regenerate_mips(self):
        """Regenerate texture mip chain from level 0 and populate Taichi field."""
        window_last = tm.ivec4(0, 0, self.width, self.height)
        window = tm.ivec4(0)
        # for _ in range(1):
        #     for x, y in ti.ndrange(self.width, self.height):
        #         self.field[window.y+y, window.x+x] = 0.5
        for _ in range(1):
            for ml in range(1, self.max_mip + 1):
                window.x = self.width 
                window.z = window.x + (self.width >> ml)
                window.y = self.height - (self.height >> tm.max(ml - 1, 0))
                window.w = window.y + (self.height >> ml)
                window_width, window_height = int(window.z - window.x), int(window.w - window.y)

                for x, y in ti.ndrange(window_width, window_height):
                    prev_addr = tm.ivec2(int(window_last.x) + (x * 2), int(window_last.y) + (y * 2))
                    avg = (self.field[prev_addr.y+0, prev_addr.x+0] + \
                        self.field[prev_addr.y+0, prev_addr.x+1] + \
                        self.field[prev_addr.y+1, prev_addr.x+0] + \
                        self.field[prev_addr.y+1, prev_addr.x+1]) * 0.25
                    self.field[window.y+y, window.x+x] = avg
                window_last = window

    # previously - ti.real_func
    @ti.func
    def _store_r(self, val:float, xy:tm.ivec2):
        self.field[xy.y, xy.x] = val

    # previously - ti.real_func
    @ti.func
    def _store_rg(self, val:tm.vec2, xy:tm.ivec2):
        self.field[xy.y, xy.x] = val.rg

    # previously - ti.real_func
    @ti.func
    def _store_rgb(self, val:tm.vec3, xy:tm.ivec2):
        self.field[xy.y, xy.x] = val.rgb

    # previously - ti.real_func
    @ti.func
    def _store_rgba(self, val:tm.vec4, xy:tm.ivec2):
        self.field[xy.y, xy.x] = val.rgba


    @ti.func
    def store(self, val:ti.template(), xy:tm.ivec2):
        """
        Store value in texture at indexed xy location.

        :param val: Value to store.
        :type val: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
        :param xy: xy index.
        :type xy: taichi.math.ivec2
        """
        if ti.static(self.channels == 1):
            self._store_r(val, xy)
        elif ti.static(self.channels == 2):
            self._store_rg(val, xy)
        elif ti.static(self.channels == 3):
            self._store_rgb(val, xy)
        elif ti.static(self.channels == 4):
            self._store_rgba(val, xy)