"""
sampler3d
=================================
Taichi 3D texture sampling module. Supports CPU, CUDA and Vulkan backends.
 """

import typing
from typing import Union

import taichi as ti
import taichi.math as tm

import torch
import numpy as np    

from .params import *

@ti.func
def sample_trilinear_clamp(tex:ti.template(), uvw:tm.vec3, repeat_w:int, repeat_h:int, repeat_d:int) -> ti.template():
    tex = tex[0, 0, 0]
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]
    eps = 1e-7

    uvw = tm.clamp(uvw, 0., 1.-eps)

    x0, y0, z0, x1, y1, z1 = 0, 0, 0, 0, 0, 0
    dim = tm.vec3(float(width * repeat_w), float(height * repeat_h),  float(depth * repeat_d))
    pos = tm.vec3(uvw.x * dim.x - 0.5, uvw.y * dim.y - 0.5, uvw.z * dim.z - 0.5)
                
    x0, y0, z0 = int(tm.floor(pos.x)), int(tm.floor(pos.y)), int(tm.floor(pos.z))
    
    dx = (pos.x + 1.) - (float(x0) + 1.)
    dy = (pos.y + 1.) - (float(y0) + 1.)
    dz = (pos.z + 1.) - (float(z0) + 1.)

    x0 = tm.clamp(x0, 0, int(dim.x)-1)%width
    y0 = tm.clamp(y0, 0, int(dim.y)-1)%height
    z0 = tm.clamp(z0, 0, int(dim.z)-1)%depth
    x1 = tm.min(x0+1, int(dim.x)-1)%width
    y1 = tm.min(y0+1, int(dim.y)-1)%height
    z1 = tm.min(z0+1, int(dim.z)-1)%depth
    
    q000 = tex[x0, y0, z0]
    q001 = tex[x0, y0, z1]
    q010 = tex[x0, y1, z0]
    q100 = tex[x1, y0, z0]
    q011 = tex[x0, y1, z1]
    q101 = tex[x1, y0, z1]
    q110 = tex[x1, y1, z0]
    q111 = tex[x1, y1, z1]
    
    q00 = tm.mix(q000, q100, dx)
    q01 = tm.mix(q001, q101, dx)
    q10 = tm.mix(q010, q110, dx)
    q11 = tm.mix(q011, q111, dx)
    
    q0 = tm.mix(q00, q01, dy)
    q1 = tm.mix(q10, q11, dy)
    
    tex = tm.mix(q0, q1, dz)
    return tex


@ti.func
def sample_trilinear_repeat(tex:ti.template(), uvw:tm.vec3, repeat_w:int, repeat_h:int, repeat_d:int) -> ti.template():
    tex = tex[0, 0, 0]
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]

    uvw = uvw % 1.

    x0, y0, z0, x1, y1, z1 = 0, 0, 0, 0, 0, 0
    dim = tm.vec3(float(width * repeat_w), float(height * repeat_h),  float(depth * repeat_d))
    pos = tm.vec3(uvw.x * dim.x - 0.5, uvw.y * dim.y - 0.5, uvw.z * dim.z - 0.5)
                
    x0, y0, z0 = int(tm.floor(pos.x)), int(tm.floor(pos.y)), int(tm.floor(pos.z))
    
    dx = (pos.x + 1.) - (float(x0) + 1.)
    dy = (pos.y + 1.) - (float(y0) + 1.)
    dz = (pos.z + 1.) - (float(z0) + 1.)

    x0, y0, z0 = x0%width, y0%height, z0%depth
    x1, y1, z1 = (x0+1)%width, (y0+1)%height, (z0+1)%depth
    
    q000 = tex[x0, y0, z0]
    q001 = tex[x0, y0, z1]
    q010 = tex[x0, y1, z0]
    q100 = tex[x1, y0, z0]
    q011 = tex[x0, y1, z1]
    q101 = tex[x1, y0, z1]
    q110 = tex[x1, y1, z0]
    q111 = tex[x1, y1, z1]
    
    q00 = tm.mix(q000, q100, dx)
    q01 = tm.mix(q001, q101, dx)
    q10 = tm.mix(q010, q110, dx)
    q11 = tm.mix(q011, q111, dx)
    
    q0 = tm.mix(q00, q01, dy)
    q1 = tm.mix(q10, q11, dy)
    
    tex = tm.mix(q0, q1, dz)
    return tex


@ti.func
def sample_nn_3d_repeat(tex:ti.template(), uvw:tm.vec3, repeat_w:int, repeat_h:int, repeat_d:int) -> ti.template():
    tex = tex[0, 0, 0]
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]
    if width > 1 or height > 1 or depth > 1:
        x, y, z = 0, 0, 0
        uvw = uvw % 1.

        x = int((uvw.x * float(width * repeat_w)) % width)
        y = int((uvw.y * float(height * repeat_h)) % height)
        z = int((uvw.z * float(depth * repeat_d)) % depth)

        tex = tex[x, y, z]
    return tex

@ti.func
def sample_nn_3d_clamp(tex:ti.template(), uvw:tm.vec3, repeat_w:int, repeat_h:int, repeat_d:int) -> ti.template():
    tex = tex[0, 0, 0]
    eps = 1e-7
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]
    if width > 1 or height > 1 or depth > 1:
        x, y, z = 0, 0, 0
        uvw = tm.clamp(uvw, 0., 1.-eps)

        x = int((uvw.x * float(width * repeat_w)) % width)
        y = int((uvw.y * float(height * repeat_h)) % height)
        z = int((uvw.z * float(depth * repeat_d)) % depth)

        tex = tex[x, y, z]
    return tex

@ti.data_oriented
class Sampler3D:
    """
    Taichi 3D texture sampler.

    :param repeat_w: Number of times to repeat image width/x.
    :param repeat_h: Number of times to repeat image height/y.
    :param repeat_d: Number of times to repeat image depth/z.
    :param filter_mode: Filter mode.
    :param wrap_mode: Wrap mode.
    """
    def __init__(self, 
        repeat_w:int=1, 
        repeat_h:int=1, 
        repeat_d:int=1, 
        filter_mode:Union[FilterMode, str]=FilterMode.TRILINEAR, 
        wrap_mode:Union[WrapMode, str]=WrapMode.REPEAT):
        self.repeat_w = repeat_w
        self.repeat_h = repeat_h
        self.repeat_d = repeat_d
        self.filter_mode = int(filter_mode) if isinstance(filter_mode, FilterMode) else FilterMode[filter_mode.strip().upper()]
        self.wrap_mode = int(wrap_mode) if isinstance(wrap_mode, WrapMode) else WrapMode[wrap_mode.strip().upper()]
        if not (self.filter_mode & FilterMode.SUPPORTED_3D):
            raise Exception("Unsupported Sampler3D filter mode")
        if not (self.wrap_mode & WrapMode.SUPPORTED_3D):
            raise Exception("Unsupported Sampler3D wrap mode")
    @ti.func
    def sample(self, tex:ti.template(), uvw:tm.vec3) -> ti.template():
        """
        Sample texture at uv coordinates.
        
        :param tex: Texture to sample.
        :type tex: Texture3D
        :param uvw: UVW coordinates.
        :type uvw: taichi.math.vec3
        :return: Filtered sampled texel.
        :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
        """
        tex = tex[0, 0, 0]
        if ti.static(self.filter_mode == FilterMode.TRILINEAR):
            if ti.static(self.wrap_mode == WrapMode.CLAMP):
                tex = sample_nn_3d_clamp(tex, uvw, self.repeat_w, self.repeat_h, self.repeat_d)
            elif ti.static(self.wrap_mode == WrapMode.REPEAT):
                tex = sample_nn_3d_repeat(tex, uvw, self.repeat_w, self.repeat_h, self.repeat_d)
        elif ti.static(self.filter_mode == FilterMode.NEAREST):
            if ti.static(self.wrap_mode == WrapMode.CLAMP):
                tex = sample_trilinear_clamp(tex, uvw, self.repeat_w, self.repeat_h, self.repeat_d)
            elif ti.static(self.wrap_mode == WrapMode.REPEAT):
                tex = sample_trilinear_repeat(tex, uvw, self.repeat_w, self.repeat_h, self.repeat_d)
        else:
            tex *= 0. # shouldn't occur; blank it so we're more likely to notice

        return tex


    @ti.func
    def _fetch_r(self, tex:ti.template(), xyz:tm.ivec3) -> float:
        return tex.field[xyz.x, xyz.y, xyz.z]

    @ti.func
    def _fetch_rg(self, tex:ti.template(), xyz:tm.ivec3) -> tm.vec2:
        return tm.vec2(tex.field[xyz.x, xyz.y, xyz.z])

    @ti.func
    def _fetch_rgb(self, tex:ti.template(), xyz:tm.ivec3) -> tm.vec3:
        return tm.vec3(tex.field[xyz.x, xyz.y, xyz.z])

    @ti.func
    def _fetch_rgba(self, tex:ti.template(), xyz:tm.ivec3) -> tm.vec4:
        return tm.vec4(tex.field[xyz.x, xyz.y, xyz.z])

    @ti.func
    def fetch(self, tex:ti.template(), xyz:tm.ivec3):
        """Fetch texel at indexed xy location.
        
        :param tex: Texture to sample.
        :type tex: Texture3D
        :param xyz: xyz index.
        :type xyz: taichi.math.ivec3
        :return: Sampled texel.
        :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
        """
        if ti.static(tex.channels == 1):
            return self._fetch_r(tex, xyz)
        if ti.static(tex.channels == 2):
            return self._fetch_rg(tex, xyz)
        if ti.static(tex.channels == 3):
            return self._fetch_rgb(tex, xyz)
        if ti.static(tex.channels == 4):
            return self._fetch_rgba(tex, xyz)