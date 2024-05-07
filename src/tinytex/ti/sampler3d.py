import typing
from typing import Union

import taichi as ti
import taichi.math as tm

import torch
import numpy as np    

from .params import *

@ti.data_oriented
class Sampler3D:
    """
    3D texture sampler.

    :param repeat_u: Number of times to tile u/width.
    :param repeat_v: Number of times to tile v/height.
    :param repeat_w: Number of times to tile w/depth.
    :param filter_mode: Filter mode.
    :param wrap_mode: Wrap mode.
    """
    def __init__(self, 
        repeat_u:int=1, 
        repeat_v:int=1, 
        repeat_w:int=1, 
        filter_mode:Union[FilterMode, str]=FilterMode.TRILINEAR, 
        wrap_mode:Union[WrapMode, str]=WrapMode.REPEAT):
        self.repeat_u = repeat_u
        self.repeat_v = repeat_v
        self.repeat_w = repeat_w
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

        if ti.static(tex.channels == 1):
            if ti.static(self.filter_mode == FilterMode.NEAREST):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_3d_nn_r_repeat(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_3d_nn_r_clamp(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_3d_nn_r_repeat_x(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_3d_nn_r_repeat_y(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Z):
                    return sample_3d_nn_r_repeat_z(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
            elif ti.static(self.filter_mode == FilterMode.TRILINEAR):
                if ti.static(self.wrap_mode == WrapMode.CLAMP):
                    tex = sample_3d_trilinear_r_clamp(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT):
                    tex = sample_3d_trilinear_r_repeat(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    tex = sample_3d_trilinear_r_repeat_x(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    tex = sample_3d_trilinear_r_repeat_y(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Z):
                    tex = sample_3d_trilinear_r_repeat_z(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)

        elif ti.static(tex.channels == 2):
            if ti.static(self.filter_mode == FilterMode.NEAREST):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_3d_nn_rg_repeat(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_3d_nn_rg_clamp(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_3d_nn_rg_repeat_x(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_3d_nn_rg_repeat_y(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Z):
                    return sample_3d_nn_rg_repeat_z(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
            elif ti.static(self.filter_mode == FilterMode.TRILINEAR):
                if ti.static(self.wrap_mode == WrapMode.CLAMP):
                    tex = sample_3d_trilinear_rg_clamp(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT):
                    tex = sample_3d_trilinear_rg_repeat(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    tex = sample_3d_trilinear_rg_repeat_x(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    tex = sample_3d_trilinear_rg_repeat_y(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Z):
                    tex = sample_3d_trilinear_rg_repeat_z(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)

        elif ti.static(tex.channels == 3):
            if ti.static(self.filter_mode == FilterMode.NEAREST):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_3d_nn_rgb_repeat(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_3d_nn_rgb_clamp(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_3d_nn_rgb_repeat_x(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_3d_nn_rgb_repeat_y(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Z):
                    return sample_3d_nn_rgb_repeat_z(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
            elif ti.static(self.filter_mode == FilterMode.TRILINEAR):
                if ti.static(self.wrap_mode == WrapMode.CLAMP):
                    tex = sample_3d_trilinear_rgb_clamp(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT):
                    tex = sample_3d_trilinear_rgb_repeat(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    tex = sample_3d_trilinear_rgb_repeat_x(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    tex = sample_3d_trilinear_rgb_repeat_y(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Z):
                    tex = sample_3d_trilinear_rgb_repeat_z(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)

        elif ti.static(tex.channels == 4):
            if ti.static(self.filter_mode == FilterMode.NEAREST):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_3d_nn_rgba_repeat(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_3d_nn_rgba_clamp(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_3d_nn_rgba_repeat_x(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_3d_nn_rgba_repeat_y(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Z):
                    return sample_3d_nn_rgba_repeat_z(tex.field, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
            elif ti.static(self.filter_mode == FilterMode.TRILINEAR):
                if ti.static(self.wrap_mode == WrapMode.CLAMP):
                    tex = sample_3d_trilinear_rgba_clamp(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT):
                    tex = sample_3d_trilinear_rgba_repeat(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    tex = sample_3d_trilinear_rgba_repeat_x(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    tex = sample_3d_trilinear_rgba_repeat_y(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Z):
                    tex = sample_3d_trilinear_rgba_repeat_z(tex, uvw, self.repeat_u, self.repeat_v, self.repeat_w)

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

@ti.func
def sample_3d_trilinear_clamp(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> ti.template():
    """
    Sample 3D field with trilinear sampling. All axes clamped.
    
    :param tex: 3D texture field.
    :type tex: taichi.field
    :param uvw: UVW coordinates.
    :type uvw: taichi.math.vec3
    :param repeat_u: Number of times to tile u/width.
    :param repeat_v: Number of times to tile v/height.
    :param repeat_w: Number of times to tile w/depth.    
    :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
    """
    tex = tex[0, 0, 0]
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]
    eps = 1e-7

    uvwb = tm.vec3(0.)
    uvwb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_u) % 1., hp.x, 1. - hp.x)
    uvwb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_v) % 1., hp.y, 1. - hp.y)
    uvwb.z = tm.clamp((tm.clamp(uv.z, 0., 1. - eps) * repeat_w) % 1., hp.z, 1. - hp.z)

    x0, y0, z0, x1, y1, z1 = 0, 0, 0, 0, 0, 0
    pos = tm.vec2(uvwb.x * width, uvwb.y * height, uvwb.z * depth)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.) 
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)
    pos.z = tm.clamp(pos.z - 0.5, 0., depth - 1.)

    x0, y0, z0 = int(pos.x), int(pos.y), int(pos.z)

    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    dz = pos.z - float(z0)

    x0, y0 = int(pos.x), int(pos.y), int(pos.z)
    x1 = tm.min(x0+1, int(width - 1))
    y1 = tm.min(y0+1, int(height - 1))
    z1 = tm.min(z0+1, int(depth - 1))
    
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
def sample_3d_trilinear_repeat(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> ti.template():
    """
    Sample 3D field with trilinear sampling. All axes repeated.
    
    :param tex: 3D texture field.
    :type tex: taichi.field
    :param uvw: UVW coordinates.
    :type uvw: taichi.math.vec3
    :param repeat_u: Number of times to tile u/width.
    :param repeat_v: Number of times to tile v/height.
    :param repeat_w: Number of times to tile w/depth.    
    :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
    """
    tex = tex[0, 0, 0]
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]

    uvwb = (uvw * tm.vec3(repeat_u, repeat_v, repeat_w)) % 1.

    x0, y0, z0, x1, y1, z1 = 0, 0, 0, 0, 0, 0
    pos = tm.vec2(uvwb.x * width, uvwb.y * height, uvwb.z * depth)
    pos.x = (pos.x - 0.5) % width
    pos.y = (pos.y - 0.5) % height
    pos.z = (pos.z - 0.5) % depth

    x0, y0, z0 = int(pos.x), int(pos.y), int(pos.z)

    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    dx = pos.z - float(z0)
    
    x1 = (x0+1) % width
    y1 = (y0+1) % height
    z1 = (z0+1) % depth
    
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
def sample_3d_trilinear_repeat_x(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> ti.template():
    """
    Sample 3D field with trilinear sampling. X-axis repeating, Y and Z clamped.
    
    :param tex: 3D texture field.
    :type tex: taichi.field
    :param uvw: UVW coordinates.
    :type uvw: taichi.math.vec3
    :param repeat_u: Number of times to tile u/width.
    :param repeat_v: Number of times to tile v/height.
    :param repeat_w: Number of times to tile w/depth.    
    :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
    """
    tex = tex[0, 0, 0]
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]

    uvwb = tm.vec3(0.)
    uvwb.x = (uvw.x * repeat_u) % 1
    uvwb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_v) % 1., hp.y, 1. - hp.y)
    uvwb.z = tm.clamp((tm.clamp(uv.z, 0., 1. - eps) * repeat_w) % 1., hp.w, 1. - hp.z)

    x0, y0, z0, x1, y1, z1 = 0, 0, 0, 0, 0, 0
    pos = tm.vec2(uvwb.x * width, uvwb.y * height, uvwb.z * depth)
    pos.x = (pos.x - 0.5) % width
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)
    pos.z = tm.clamp(pos.z - 0.5, 0., depth - 1.)

    x0, y0, z0 = int(pos.x), int(pos.y), int(pos.z)

    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    dx = pos.z - float(z0)
    
    x1 = (x0+1) % width
    y1 = tm.min(y0+1, int(height - 1))
    z1 = tm.min(z0+1, int(depth - 1))
    
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
def sample_3d_trilinear_repeat_y(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> ti.template():
    """
    Sample 3D field with trilinear sampling. Y-axis repeating, X and Z clamped.
    
    :param tex: 3D texture field.
    :type tex: taichi.field
    :param uvw: UVW coordinates.
    :type uvw: taichi.math.vec3
    :param repeat_u: Number of times to tile u/width.
    :param repeat_v: Number of times to tile v/height.
    :param repeat_w: Number of times to tile w/depth.    
    :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
    """
    tex = tex[0, 0, 0]
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]

    uvwb = tm.vec3(0.)
    uvwb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_u) % 1., hp.x, 1. - hp.x)
    uvwb.y = (uvw.y * repeat_v) % 1
    uvwb.z = tm.clamp((tm.clamp(uv.z, 0., 1. - eps) * repeat_w) % 1., hp.z, 1. - hp.z)

    x0, y0, z0, x1, y1, z1 = 0, 0, 0, 0, 0, 0
    pos = tm.vec2(uvwb.x * width, uvwb.y * height, uvwb.z * depth)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.) 
    pos.y = (pos.y - 0.5) % height
    pos.z = tm.clamp(pos.z - 0.5, 0., depth - 1.)

    x0, y0, z0 = int(pos.x), int(pos.y), int(pos.z)

    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    dx = pos.z - float(z0)
    
    x1 = tm.min(x0+1, int(width - 1))
    y1 = (y0+1) % height
    z1 = tm.min(z0+1, int(depth - 1))    

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
def sample_3d_trilinear_repeat_z(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> ti.template():
    """
    Sample 3D field with trilinear sampling. Z-axis repeating, X and Y clamped.
    
    :param tex: 3D texture field.
    :type tex: taichi.field
    :param uvw: UVW coordinates.
    :type uvw: taichi.math.vec3
    :param repeat_u: Number of times to tile u/width.
    :param repeat_v: Number of times to tile v/height.
    :param repeat_w: Number of times to tile w/depth.    
    :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
    """
    tex = tex[0, 0, 0]
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]

    uvwb = tm.vec3(0.)
    uvwb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_u) % 1., hp.x, 1. - hp.x)
    uvwb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_v) % 1., hp.y, 1. - hp.y)
    uvwb.z = (uvw.x * repeat_u) % 1

    x0, y0, z0, x1, y1, z1 = 0, 0, 0, 0, 0, 0
    pos = tm.vec2(uvwb.x * width, uvwb.y * height, uvwb.z * depth)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.) 
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)
    pos.z = (pos.z - 0.5) % depth

    x0, y0, z0 = int(pos.x), int(pos.y), int(pos.z)

    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    dx = pos.z - float(z0)
    
    x1 = tm.min(x0+1, int(width - 1))
    y1 = tm.min(y0+1, int(height - 1))
    z1 = (z0+1) % depth
    
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


# previously - ti.real_func
@ti.func
def sample_3d_trilinear_r_repeat(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> float:
    return sample_3d_trilinear_repeat(tex, uvw, repeat_u, repeat_v, repeat_w)

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_r_clamp(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> float:
    return sample_3d_trilinear_clamp(tex, uvw, repeat_u, repeat_v, repeat_w)

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_r_repeat_x(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> float:
    return sample_3d_trilinear_repeat_x(tex, uvw, repeat_u, repeat_v, repeat_w)

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_r_repeat_y(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> float:
    return sample_3d_trilinear_repeat_y(tex, uvw, repeat_u, repeat_v, repeat_w)

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_r_repeat_z(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> float:
    return sample_3d_trilinear_repeat_z(tex, uvw, repeat_u, repeat_v, repeat_w)


# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rg_repeat(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec2:
    return tm.vec2(sample_3d_trilinear_repeat(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rg_clamp(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec2:
    return tm.vec2(sample_3d_trilinear_clamp(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rg_repeat_x(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec2:
    return tm.vec2(sample_3d_trilinear_repeat_x(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rg_repeat_y(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec2:
    return tm.vec2(sample_3d_trilinear_repeat_y(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rg_repeat_z(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec2:
    return tm.vec2(sample_3d_trilinear_repeat_z(tex, uvw, repeat_u, repeat_v, repeat_w))


# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rgb_repeat(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec3:
    return tm.vec3(sample_3d_trilinear_repeat(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rgb_clamp(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec3:
    return tm.vec3(sample_3d_trilinear_clamp(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rgb_repeat_x(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec3:
    return tm.vec3(sample_3d_trilinear_repeat_x(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rgb_repeat_y(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec3:
    return tm.vec3(sample_3d_trilinear_repeat_y(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rgb_repeat_z(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec3:
    return tm.vec3(sample_3d_trilinear_repeat_z(tex, uvw, repeat_u, repeat_v, repeat_w))


# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rgba_repeat(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec4:
    return tm.vec4(sample_3d_trilinear_repeat(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rgba_clamp(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec4:
    return tm.vec4(sample_3d_trilinear_clamp(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rgba_repeat_x(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec4:
    return tm.vec4(sample_3d_trilinear_repeat_x(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rgba_repeat_y(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec4:
    return tm.vec4(sample_3d_trilinear_repeat_y(tex, uvw, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_trilinear_rgba_repeat_z(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> tm.vec4:
    return tm.vec4(sample_3d_trilinear_repeat_z(tex, uvw, repeat_u, repeat_v, repeat_w))



@ti.func
def sample_3d_nn_repeat(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> ti.template():
    """
    Sample 3D field with point sampling. All axes repeating.
    
    :param tex: 3D texture field.
    :type tex: taichi.field
    :param uvw: UVW coordinates.
    :type uvw: taichi.math.vec3
    :param repeat_u: Number of times to tile u/width.
    :param repeat_v: Number of times to tile v/height.
    :param repeat_w: Number of times to tile w/depth.    
    :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
    """
    tex = tex[0, 0, 0]
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]
    if width > 1 or height > 1 or depth > 1:
        x, y, z = 0, 0, 0
        uvwb = uvw % 1.

        x = int((uvwb.x * float(width * repeat_u)) % width)
        y = int((uvwb.y * float(height * repeat_v)) % height)
        z = int((uvwb.z * float(depth * repeat_w)) % depth)

        tex = tex[x, y, z]
    return tex

@ti.func
def sample_3d_nn_clamp(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> ti.template():
    """
    Sample 3D field with point sampling. All axes clamped.
    
    :param tex: 3D texture field.
    :type tex: taichi.field
    :param uvw: UVW coordinates.
    :type uvw: taichi.math.vec3
    :param repeat_u: Number of times to tile u/width.
    :param repeat_v: Number of times to tile v/height.
    :param repeat_w: Number of times to tile w/depth.    
    :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
    """
    tex = tex[0, 0, 0]
    eps = 1e-7
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]
    if width > 1 or height > 1 or depth > 1:
        x, y, z = 0, 0, 0
        uvwb = tm.clamp(uvw, 0., 1.-eps)

        x = int((uvwb.x * float(width * repeat_u)) % width)
        y = int((uvwb.y * float(height * repeat_v)) % height)
        z = int((uvwb.z * float(depth * repeat_w)) % depth)

        tex = tex[x, y, z]
    return tex


@ti.func
def sample_3d_nn_repeat_x(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> ti.template():
    """
    Sample 3D field with point sampling. All axes repeating.
    
    :param tex: 3D texture field.
    :type tex: taichi.field
    :param uvw: UVW coordinates.
    :type uvw: taichi.math.vec3
    :param repeat_u: Number of times to tile u/width.
    :param repeat_v: Number of times to tile v/height.
    :param repeat_w: Number of times to tile w/depth.    
    :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
    """
    tex = tex[0, 0, 0]
    eps = 1e-7
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]
    if width > 1 or height > 1 or depth > 1:
        x, y, z = 0, 0, 0

        uvwb = tm.vec3(0.)
        uvwb.x = uvw.x % 1.
        uvwb.y = tm.clamp(uvw.y, 0., 1.-eps)
        uvwb.z = tm.clamp(uvw.z, 0., 1.-eps)

        x = int((uvwb.x * float(width * repeat_u)) % width)
        y = int((uvwb.y * float(height * repeat_v)) % height)
        z = int((uvwb.z * float(depth * repeat_w)) % depth)

        tex = tex[x, y, z]
    return tex

@ti.func
def sample_3d_nn_repeat_y(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> ti.template():
    """
    Sample 3D field with point sampling. All axes repeating.
    
    :param tex: 3D texture field.
    :type tex: taichi.field
    :param uvw: UVW coordinates.
    :type uvw: taichi.math.vec3
    :param repeat_u: Number of times to tile u/width.
    :param repeat_v: Number of times to tile v/height.
    :param repeat_w: Number of times to tile w/depth.    
    :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
    """
    tex = tex[0, 0, 0]
    eps = 1e-7
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]
    if width > 1 or height > 1 or depth > 1:
        x, y, z = 0, 0, 0

        uvwb = tm.vec3(0.)
        uvwb.x = tm.clamp(uvw.x, 0., 1.-eps)
        uvwb.y = uvw.y % 1.
        uvwb.z = tm.clamp(uvw.z, 0., 1.-eps)

        x = int((uvwb.x * float(width * repeat_u)) % width)
        y = int((uvwb.y * float(height * repeat_v)) % height)
        z = int((uvwb.z * float(depth * repeat_w)) % depth)

        tex = tex[x, y, z]
    return tex

@ti.func
def sample_3d_nn_repeat_z(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int, repeat_w:int) -> ti.template():
    """
    Sample 3D field with point sampling. All axes repeating.

    :param tex: 3D texture field.
    :type tex: taichi.field
    :param uvw: UVW coordinates.
    :type uvw: taichi.math.vec3
    :param repeat_u: Number of times to tile u/width.
    :param repeat_v: Number of times to tile v/height.
    :param repeat_w: Number of times to tile w/depth.
    :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
    """
    tex = tex[0, 0, 0]
    eps = 1e-7
    width, height, depth = tex.shape[0], tex.shape[1], tex.shape[2]
    if width > 1 or height > 1 or depth > 1:
        x, y, z = 0, 0, 0

        uvwb = tm.vec3(0.)
        uvwb.x = tm.clamp(uvw.x, 0., 1.-eps)
        uvwb.y = tm.clamp(uvw.y, 0., 1.-eps)
        uvwb.z = uvw.z % 1.

        x = int((uvwb.x * float(width * repeat_u)) % width)
        y = int((uvwb.y * float(height * repeat_v)) % height)
        z = int((uvwb.z * float(depth * repeat_w)) % depth)

        tex = tex[x, y, z]
    return tex


# previously - ti.real_func
@ti.func
def sample_3d_nn_r_repeat(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> float:
    return sample_3d_nn_repeat(tex, uv, repeat_u, repeat_v, repeat_w)

# previously - ti.real_func
@ti.func
def sample_3d_nn_r_clamp(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> float:
    return sample_3d_nn_clamp(tex, uv, repeat_u, repeat_v, repeat_w)

# previously - ti.real_func
@ti.func
def sample_3d_nn_r_repeat_x(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> float:
    return sample_3d_nn_repeat_x(tex, uv, repeat_u, repeat_v, repeat_w)

# previously - ti.real_func
@ti.func
def sample_3d_nn_r_repeat_y(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> float:
    return sample_3d_nn_repeat_y(tex, uv, repeat_u, repeat_v, repeat_w)

# previously - ti.real_func
@ti.func
def sample_3d_nn_r_repeat_z(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> float:
    return sample_3d_nn_repeat_z(tex, uv, repeat_u, repeat_v, repeat_w)



# previously - ti.real_func
@ti.func
def sample_3d_nn_rg_repeat(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec2:
    return tm.vec2(sample_3d_nn_repeat(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rg_clamp(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec2:
    return tm.vec2(sample_3d_nn_clamp(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rg_repeat_x(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec2:
    return tm.vec2(sample_3d_nn_repeat_x(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rg_repeat_y(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec2:
    return tm.vec2(sample_3d_nn_repeat_y(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rg_repeat_z(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec2:
    return tm.vec2(sample_3d_nn_repeat_z(tex, uv, repeat_u, repeat_v, repeat_w))



# previously - ti.real_func
@ti.func
def sample_3d_nn_rgb_repeat(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec3:
    return tm.vec3(sample_3d_nn_repeat(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rgb_clamp(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec3:
    return tm.vec3(sample_3d_nn_clamp(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rgb_repeat_x(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec3:
    return tm.vec3(sample_3d_nn_repeat_x(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rgb_repeat_y(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec3:
    return tm.vec3(sample_3d_nn_repeat_y(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rgb_repeat_z(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec3:
    return tm.vec3(sample_3d_nn_repeat_z(tex, uv, repeat_u, repeat_v, repeat_w))


# previously - ti.real_func
@ti.func
def sample_3d_nn_rgba_repeat(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec4:
    return tm.vec4(sample_3d_nn_repeat(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rgba_clamp(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec4:
    return tm.vec4(sample_3d_nn_clamp(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rgba_repeat_x(tex:ti.template(), uvw:tm.vec3, repeat_u:int, repeat_v:int,  repeat_w:int) -> tm.vec4:
    return tm.vec4(sample_3d_nn_repeat_x(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rgba_repeat_y(tex:ti.template(), uv:tm.vec2, repeat_u:int, repeat_v:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_3d_nn_repeat_y(tex, uv, repeat_u, repeat_v, repeat_w))

# previously - ti.real_func
@ti.func
def sample_3d_nn_rgba_repeat_z(tex:ti.template(), uv:tm.vec2, repeat_u:int, repeat_v:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_3d_nn_repeat_z(tex, uv, repeat_u, repeat_v, repeat_w))