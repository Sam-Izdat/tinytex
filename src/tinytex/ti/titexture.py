"""
titexture
=================================
Taichi texture sampling module. Supports CPU, CUDA and Vulkan backends.
 """

import typing
from typing import Union

import taichi as ti
import taichi.math as tm

import torch
import numpy as np
from enum import IntEnum    

class FilterMode(IntEnum):
    """Texture filter mode"""
    NEAREST             = 1<<0
    BILINEAR            = 1<<1
    TRILINEAR           = 1<<2
    BICUBIC             = 1<<3
    B_SPLINE            = 1<<4
    MITCHELL_NETRAVALI  = 1<<5
    CATMULL_ROM         = 1<<6

    SUPPORTED_2D = NEAREST | BILINEAR | BICUBIC | B_SPLINE | MITCHELL_NETRAVALI | CATMULL_ROM
    SUPPORTED_3D = NEAREST | TRILINEAR

class WrapMode(IntEnum):
    """Texture wrap mode"""
    REPEAT      = 1<<0
    CLAMP       = 1<<1
    REPEAT_X    = 1<<2
    REPEAT_Y    = 1<<3

    # TODO: MIRROR, etc

    SUPPORTED_2D = REPEAT | CLAMP | REPEAT_X | REPEAT_Y
    SUPPORTED_3D = REPEAT | CLAMP | REPEAT_X | REPEAT_Y

@ti.func
def interpolant_cubic_spline(p:tm.vec4, x:float) -> float:
    return p[1] + 0.5 * x*(p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0])))

@ti.func
def interpolant_b_spline(p:tm.vec4, x:float) -> float:
    third = (1./3.)
    sixth = (1./6.)
    x_squared = x**2
    x_cubed = x**3
    out = (-sixth * p[0] + 0.5 * p[1] - 0.5 * p[2] + sixth * p[3]) * x_cubed \
        + (0.5 * p[0] - 1. * p[1] + 0.5 * p[2]) * x_squared \
        + (-0.5 * p[0] + 0.5 * p[2]) * x \
        + sixth * p[0] + (-third + 1.) * p[1] + (sixth * p[2])
    return out

@ti.func
def interpolant_mitchell_netravali(p:tm.vec4, x:float, b:float, c:float) -> float:
    third = (1./3.)
    sixth = (1./6.)
    B, C = b, c
    x_squared = x**2
    x_cubed = x**3
    out = ((-sixth * B - C) * p[0] + (-(3./2.) * B - C + 2.) * p[1] + ((3./2.) * B + C - 2.) * p[2] + (sixth * B + C) * p[3]) * x_cubed \
        + ((0.5 * B + 2. * C) * p[0] + (2. * B + C - 3.) * p[1] + ((-5./2.) * B - 2 * C + 3.) * p[2] - C * p[3]) * x_squared \
        + ((-0.5 * B - C) * p[0] + (0.5 * B + C) * p[2]) * x \
        + (sixth * B) * p[0] + (-third * B + 1.) * p[1] + (sixth * B * p[2])
    return out

@ti.func
def spline_cubic(p:tm.mat4, x:float, y:float) -> float:
    arr = tm.vec4(0.)
    arr[0] = interpolant_cubic_spline(tm.vec4(p[0,:]), y)
    arr[1] = interpolant_cubic_spline(tm.vec4(p[1,:]), y)
    arr[2] = interpolant_cubic_spline(tm.vec4(p[2,:]), y)
    arr[3] = interpolant_cubic_spline(tm.vec4(p[3,:]), y)
    return interpolant_cubic_spline(arr, x)

@ti.func
def spline_b_spline(p:tm.mat4, x:float, y:float) -> float:
    arr = tm.vec4(0.)
    arr[0] = interpolant_b_spline(tm.vec4(p[0,:]), y)
    arr[1] = interpolant_b_spline(tm.vec4(p[1,:]), y)
    arr[2] = interpolant_b_spline(tm.vec4(p[2,:]), y)
    arr[3] = interpolant_b_spline(tm.vec4(p[3,:]), y)
    return interpolant_b_spline(arr, x)

@ti.func
def spline_mitchell_netravali(p:tm.mat4, x:float, y:float, b:float, c:float) -> float:
    arr = tm.vec4(0.)
    arr[0] = interpolant_mitchell_netravali(tm.vec4(p[0,:]), y, b, c)
    arr[1] = interpolant_mitchell_netravali(tm.vec4(p[1,:]), y, b, c)
    arr[2] = interpolant_mitchell_netravali(tm.vec4(p[2,:]), y, b, c)
    arr[3] = interpolant_mitchell_netravali(tm.vec4(p[3,:]), y, b, c)
    return interpolant_mitchell_netravali(arr, x, b, c)


# ------------------------------------------------------

@ti.func
def sample_cubic_repeat_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0, y0 = (x1-1) % width, (y1-1) % height
    x2, y2 = (x1+1) % width, (y1+1) % height
    x3, y3 = (x1+2) % width, (y1+2) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)
    # mat 
    # 00 01 02 03
    # 10 11 12 13
    # 20 21 22 23
    # 30 31 32 33
    #
    # n is tex.n on vector fields but does not exist for float
    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_cubic(p, dx, dy), 0.)

@ti.func
def sample_cubic_repeat_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0, y0 = (x1-1) % width, (y1-1) % height
    x2, y2 = (x1+1) % width, (y1+1) % height
    x3, y3 = (x1+2) % width, (y1+2) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_cubic(p, dx, dy), 0.)

    return out

@ti.func
def sample_cubic_clamp_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = tm.min(y1-1, int(height - 1))
    x2 = tm.min(x1+1, int(width - 1))
    y2 = tm.min(y1+1, int(height - 1))
    x3 = tm.min(x1+2, int(width - 1))
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)

    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_cubic(p, dx, dy), 0.)

    return out

@ti.func
def sample_cubic_clamp_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = tm.min(y1-1, int(height - 1))
    x2 = tm.min(x1+1, int(width - 1))
    y2 = tm.min(y1+1, int(height - 1))
    x3 = tm.min(x1+2, int(width - 1))
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_cubic(p, dx, dy), 0.)

    return out

@ti.func
def sample_cubic_repeat_x_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+3) % width
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)

    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_cubic(p, dx, dy), 0.)

    return out

@ti.func
def sample_cubic_repeat_x_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+3) % width
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_cubic(p, dx, dy), 0.)

    return out

@ti.func
def sample_cubic_repeat_y_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+1) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)
    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_cubic(p, dx, dy), 0.)

    return out

@ti.func
def sample_cubic_repeat_y_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+1) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_cubic(p, dx, dy), 0.)

    return out

# previously - ti.real_func
@ti.func
def sample_cubic_r_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_cubic_repeat_float(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_cubic_r_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_cubic_clamp_float(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_cubic_r_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_cubic_repeat_x_float(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_cubic_r_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_cubic_repeat_y_float(tex, uv, repeat_w, repeat_h, window)



# previously - ti.real_func
@ti.func
def sample_cubic_rg_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_cubic_repeat_vec(tex, uv, repeat_w, repeat_h, window, 2))

# previously - ti.real_func
@ti.func
def sample_cubic_rg_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_cubic_clamp_vec(tex, uv, repeat_w, repeat_h, window, 2))

# previously - ti.real_func
@ti.func
def sample_cubic_rg_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_cubic_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 2))

# previously - ti.real_func
@ti.func
def sample_cubic_rg_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_cubic_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 2))



# previously - ti.real_func
@ti.func
def sample_cubic_rgb_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_cubic_repeat_vec(tex, uv, repeat_w, repeat_h, window, 3))

# previously - ti.real_func
@ti.func
def sample_cubic_rgb_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_cubic_clamp_vec(tex, uv, repeat_w, repeat_h, window, 3))

# previously - ti.real_func
@ti.func
def sample_cubic_rgb_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_cubic_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 3))

# previously - ti.real_func
@ti.func
def sample_cubic_rgb_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_cubic_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 3))


# previously - ti.real_func
@ti.func
def sample_cubic_rgba_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_cubic_repeat_vec(tex, uv, repeat_w, repeat_h, window, 4))

# previously - ti.real_func
@ti.func
def sample_cubic_rgba_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_cubic_clamp_vec(tex, uv, repeat_w, repeat_h, window, 4))

# previously - ti.real_func
@ti.func
def sample_cubic_rgba_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_cubic_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 4))

# previously - ti.real_func
@ti.func
def sample_cubic_rgba_repeat_y(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_cubic_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 4))






@ti.func
def sample_b_spline_repeat_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0, y0 = (x1-1) % width, (y1-1) % height
    x2, y2 = (x1+1) % width, (y1+1) % height
    x3, y3 = (x1+2) % width, (y1+2) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)
    # mat 
    # 00 01 02 03
    # 10 11 12 13
    # 20 21 22 23
    # 30 31 32 33
    #
    # n is tex.n on vector fields but does not exist for float
    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_b_spline(p, dx, dy), 0.)

@ti.func
def sample_b_spline_repeat_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0, y0 = (x1-1) % width, (y1-1) % height
    x2, y2 = (x1+1) % width, (y1+1) % height
    x3, y3 = (x1+2) % width, (y1+2) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_clamp_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = tm.min(y1-1, int(height - 1))
    x2 = tm.min(x1+1, int(width - 1))
    y2 = tm.min(y1+1, int(height - 1))
    x3 = tm.min(x1+2, int(width - 1))
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)

    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_clamp_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = tm.min(y1-1, int(height - 1))
    x2 = tm.min(x1+1, int(width - 1))
    y2 = tm.min(y1+1, int(height - 1))
    x3 = tm.min(x1+2, int(width - 1))
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_repeat_x_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+3) % width
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)

    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_repeat_x_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+3) % width
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_repeat_y_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+1) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)
    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_repeat_y_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+1) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_b_spline(p, dx, dy), 0.)

    return out

# previously - ti.real_func
@ti.func
def sample_b_spline_r_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_b_spline_repeat_float(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_b_spline_r_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_b_spline_clamp_float(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_b_spline_r_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_b_spline_repeat_x_float(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_b_spline_r_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_b_spline_repeat_y_float(tex, uv, repeat_w, repeat_h, window)



# previously - ti.real_func
@ti.func
def sample_b_spline_rg_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_b_spline_repeat_vec(tex, uv, repeat_w, repeat_h, window, 2))

# previously - ti.real_func
@ti.func
def sample_b_spline_rg_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_b_spline_clamp_vec(tex, uv, repeat_w, repeat_h, window, 2))

# previously - ti.real_func
@ti.func
def sample_b_spline_rg_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_b_spline_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 2))

# previously - ti.real_func
@ti.func
def sample_b_spline_rg_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_b_spline_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 2))



# previously - ti.real_func
@ti.func
def sample_b_spline_rgb_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_b_spline_repeat_vec(tex, uv, repeat_w, repeat_h, window, 3))

# previously - ti.real_func
@ti.func
def sample_b_spline_rgb_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_b_spline_clamp_vec(tex, uv, repeat_w, repeat_h, window, 3))

# previously - ti.real_func
@ti.func
def sample_b_spline_rgb_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_b_spline_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 3))

# previously - ti.real_func
@ti.func
def sample_b_spline_rgb_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_b_spline_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 3))


# previously - ti.real_func
@ti.func
def sample_b_spline_rgba_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_b_spline_repeat_vec(tex, uv, repeat_w, repeat_h, window, 4))

# previously - ti.real_func
@ti.func
def sample_b_spline_rgba_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_b_spline_clamp_vec(tex, uv, repeat_w, repeat_h, window, 4))

# previously - ti.real_func
@ti.func
def sample_b_spline_rgba_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_b_spline_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 4))

# previously - ti.real_func
@ti.func
def sample_b_spline_rgba_repeat_y(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_b_spline_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 4))

















@ti.func
def sample_mitchell_netravali_repeat_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0, y0 = (x1-1) % width, (y1-1) % height
    x2, y2 = (x1+1) % width, (y1+1) % height
    x3, y3 = (x1+2) % width, (y1+2) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)
    # mat 
    # 00 01 02 03
    # 10 11 12 13
    # 20 21 22 23
    # 30 31 32 33
    #
    # n is tex.n on vector fields but does not exist for float
    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_mitchell_netravali(p, dx, dy, b, c), 0.)

@ti.func
def sample_mitchell_netravali_repeat_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0, y0 = (x1-1) % width, (y1-1) % height
    x2, y2 = (x1+1) % width, (y1+1) % height
    x3, y3 = (x1+2) % width, (y1+2) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

@ti.func
def sample_mitchell_netravali_clamp_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = tm.min(y1-1, int(height - 1))
    x2 = tm.min(x1+1, int(width - 1))
    y2 = tm.min(y1+1, int(height - 1))
    x3 = tm.min(x1+2, int(width - 1))
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)

    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_mitchell_netravali(p, dx, dy, b, c), 0.)
    return out

@ti.func
def sample_mitchell_netravali_clamp_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = tm.min(y1-1, int(height - 1))
    x2 = tm.min(x1+1, int(width - 1))
    y2 = tm.min(y1+1, int(height - 1))
    x3 = tm.min(x1+2, int(width - 1))
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

@ti.func
def sample_mitchell_netravali_repeat_x_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+3) % width
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)

    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

@ti.func
def sample_mitchell_netravali_repeat_x_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = tm.clamp(pos.y, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+3) % width
    y3 = tm.min(y1+2, int(height - 1))

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

@ti.func
def sample_mitchell_netravali_repeat_y_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+1) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4(0.)
    for ch in range(n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(spline_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

@ti.func
def sample_mitchell_netravali_repeat_y_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = pos.y % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+1) % height

    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x1)
    dy = pos.y - float(y1)

    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q02 = tex[yofs + y2, xofs + x0]
    q03 = tex[yofs + y3, xofs + x0]

    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]
    q12 = tex[yofs + y2, xofs + x1]
    q13 = tex[yofs + y3, xofs + x1]

    q20 = tex[yofs + y0, xofs + x2]
    q21 = tex[yofs + y1, xofs + x2]
    q22 = tex[yofs + y2, xofs + x2]
    q23 = tex[yofs + y3, xofs + x2]

    q30 = tex[yofs + y0, xofs + x3]
    q31 = tex[yofs + y1, xofs + x3]
    q32 = tex[yofs + y2, xofs + x3]
    q33 = tex[yofs + y3, xofs + x3]

    out = q11
    p = tm.mat4([
        [q00, q01, q02, q03],
        [q10, q11, q12, q13],
        [q20, q21, q22, q23],
        [q30, q31, q32, q33]
        ])        
    out = tm.max(spline_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_r_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> float:
    return sample_mitchell_netravali_repeat_float(tex, uv, repeat_w, repeat_h, window, b, c)

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_r_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> float:
    return sample_mitchell_netravali_clamp_float(tex, uv, repeat_w, repeat_h, window, b, c)

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_r_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> float:
    return sample_mitchell_netravali_repeat_x_float(tex, uv, repeat_w, repeat_h, window, b, c)

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_r_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> float:
    return sample_mitchell_netravali_repeat_y_float(tex, uv, repeat_w, repeat_h, window, b, c)



# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rg_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec2:
    return tm.vec2(sample_mitchell_netravali_repeat_vec(tex, uv, repeat_w, repeat_h, window, 2, b, c))

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rg_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec2:
    return tm.vec2(sample_mitchell_netravali_clamp_vec(tex, uv, repeat_w, repeat_h, window, 2, b, c))

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rg_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec2:
    return tm.vec2(sample_mitchell_netravali_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 2, b, c))

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rg_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec2:
    return tm.vec2(sample_mitchell_netravali_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 2, b, c))



# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rgb_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec3:
    return tm.vec3(sample_mitchell_netravali_repeat_vec(tex, uv, repeat_w, repeat_h, window, 3, b, c))

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rgb_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec3:
    return tm.vec3(sample_mitchell_netravali_clamp_vec(tex, uv, repeat_w, repeat_h, window, 3, b, c))

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rgb_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec3:
    return tm.vec3(sample_mitchell_netravali_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 3, b, c))

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rgb_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec3:
    return tm.vec3(sample_mitchell_netravali_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 3, b, c))


# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rgba_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec4:
    return tm.vec4(sample_mitchell_netravali_repeat_vec(tex, uv, repeat_w, repeat_h, window, 4, b, c))

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rgba_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec4:
    return tm.vec4(sample_mitchell_netravali_clamp_vec(tex, uv, repeat_w, repeat_h, window, 4, b, c))

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rgba_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec4:
    return tm.vec4(sample_mitchell_netravali_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 4, b, c))

# previously - ti.real_func
@ti.func
def sample_mitchell_netravali_rgba_repeat_y(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float) -> tm.vec4:
    return tm.vec4(sample_mitchell_netravali_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 4, b, c))







@ti.func
def sample_nn_repeat(
    tex:ti.template(), 
    uv:tm.vec2, 
    wrap_mode:int, 
    repeat_w:int, 
    repeat_h:int,  
    window:tm.ivec4) -> ti.template():
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = uv
    x, y = 0, 0
    uvb.x = uv.x % 1.
    uvb.y = uv.y % 1.
    x = int(window.x + ((uvb.x * float(width * repeat_w)) % width))
    y = int(window.y + ((uvb.y * float(height * repeat_h)) % height))    
    return tex[y, x]

@ti.func
def sample_nn_clamp(
    tex:ti.template(), 
    uv:tm.vec2, 
    wrap_mode:int, 
    repeat_w:int, 
    repeat_h:int,  
    window:tm.ivec4) -> ti.template():
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = uv
    x, y = 0, 0
    uvb.x = tm.clamp(uv.x, 0., 1. - (0.5 / width)) 
    uvb.y = tm.clamp(uv.y, 0., 1. - (0.5 / height))
    x = int(window.x + ((uvb.x * float(width * repeat_w)) % width))
    y = int(window.y + ((uvb.y * float(height * repeat_h)) % height))    
    return tex[y, x]

@ti.func
def sample_nn_repeat_x(
    tex:ti.template(), 
    uv:tm.vec2, 
    wrap_mode:int, 
    repeat_w:int, 
    repeat_h:int,  
    window:tm.ivec4) -> ti.template():
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = uv
    x, y = 0, 0
    uvb.x = uv.x % 1.
    uvb.y = tm.clamp(uv.y, 0., 1. - (0.5 / height))
    x = int(window.x + ((uvb.x * float(width * repeat_w)) % width))
    y = int(window.y + ((uvb.y * float(height * repeat_h)) % height))    
    return tex[y, x]

@ti.func
def sample_nn_repeat_y(
    tex:ti.template(), 
    uv:tm.vec2, 
    wrap_mode:int, 
    repeat_w:int, 
    repeat_h:int,  
    window:tm.ivec4) -> ti.template():
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = uv
    x, y = 0, 0
    uvb.x = tm.clamp(uv.x, 0., 1. - (0.5 / width)) 
    uvb.y = uv.y % 1.
    x = int(window.x + ((uvb.x * float(width * repeat_w)) % width))
    y = int(window.y + ((uvb.y * float(height * repeat_h)) % height))    
    return tex[y, x]


# previously - ti.real_func
@ti.func
def sample_nn_r_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_nn_repeat(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_nn_r_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_nn_clamp(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_nn_r_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_nn_repeat_x(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_nn_r_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_nn_repeat_y(tex, uv, repeat_w, repeat_h, window)



# previously - ti.real_func
@ti.func
def sample_nn_rg_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_nn_repeat(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_nn_rg_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_nn_clamp(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_nn_rg_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_nn_repeat_x(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_nn_rg_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_nn_repeat_y(tex, uv, repeat_w, repeat_h, window))



# previously - ti.real_func
@ti.func
def sample_nn_rgb_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_nn_repeat(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgb_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_nn_clamp(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgb_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_nn_repeat_x(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgb_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_nn_repeat_y(tex, uv, repeat_w, repeat_h, window))


# previously - ti.real_func
@ti.func
def sample_nn_rgba_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_nn_repeat(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgba_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_nn_clamp(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgba_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_nn_repeat_x(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgba_repeat_y(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_nn_repeat_y(tex, uv, repeat_w, repeat_h, window))


@ti.func
def sample_bilinear_repeat(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = pos.y % height
    x0, y0 = int(pos.x), int(pos.y)
    x1 = (x0+1) % width
    y1 = (y0+1) % height
    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]        
    q0 = tm.mix(q00, q10, dx)
    q1 = tm.mix(q01, q11, dx)
    out = tm.mix(q0, q1, dy)
    return out

@ti.func
def sample_bilinear_clamp(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = tm.clamp(pos.y, 0., height - 1.)
    x0, y0 = int(pos.x), int(pos.y)
    x1 = tm.min(x0+1, int(width - 1))
    y1 = tm.min(y0+1, int(height - 1))
    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]        
    q0 = tm.mix(q00, q10, dx)
    q1 = tm.mix(q01, q11, dx)
    out = tm.mix(q0, q1, dy)
    return out

@ti.func
def sample_bilinear_repeat_x(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hpy = 0.5 / height
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hpy, 1. - hpy) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = tm.clamp(pos.y, 0., height - 1.)
    x0, y0 = int(pos.x), int(pos.y)
    x1 = (x0+1) % width
    y1 = tm.min(y0+1, int(height - 1))
    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]        
    q0 = tm.mix(q00, q10, dx)
    q1 = tm.mix(q01, q11, dx)
    out = tm.mix(q0, q1, dy)
    return out

@ti.func
def sample_bilinear_repeat_y(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hpx = 0.5 / width
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hpx, 1. - hpx) * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = pos.y % height
    x0, y0 = int(pos.x), int(pos.y)
    x1 = tm.min(x0+1, int(width - 1))
    y1 = (y0+1) % height    
    xofs, yofs = int(window.x), int(window.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    q00 = tex[yofs + y0, xofs + x0]
    q01 = tex[yofs + y1, xofs + x0]
    q10 = tex[yofs + y0, xofs + x1]
    q11 = tex[yofs + y1, xofs + x1]        
    q0 = tm.mix(q00, q10, dx)
    q1 = tm.mix(q01, q11, dx)
    out = tm.mix(q0, q1, dy)
    return out


# previously - ti.real_func
@ti.func
def sample_bilinear_r_repeat(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_bilinear_repeat(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_bilinear_r_clamp(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_bilinear_clamp(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_bilinear_r_repeat_x(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_bilinear_repeat_x(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_bilinear_r_repeat_y(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_bilinear_repeat_y(tex, uv, repeat_w, repeat_h, window)


# previously - ti.real_func
@ti.func
def sample_bilinear_rg_repeat(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_bilinear_repeat(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rg_clamp(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_bilinear_clamp(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rg_repeat_x(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_bilinear_repeat_x(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rg_repeat_y(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_bilinear_repeat_y(tex, uv, repeat_w, repeat_h, window))


# previously - ti.real_func
@ti.func
def sample_bilinear_rgb_repeat(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_bilinear_repeat(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgb_clamp(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_bilinear_clamp(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgb_repeat_x(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_bilinear_repeat_x(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgb_repeat_y(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_bilinear_repeat_y(tex, uv, repeat_w, repeat_h, window))


# previously - ti.real_func
@ti.func
def sample_bilinear_rgba_repeat(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_bilinear_repeat(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgba_clamp(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_bilinear_clamp(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgba_repeat_x(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_bilinear_repeat_x(tex, uv, repeat_w, repeat_h, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgba_repeat_y(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_bilinear_repeat_y(tex, uv, repeat_w, repeat_h, window))




@ti.func
def _sample_bilinear_clamp_partial(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    hp = tm.vec2(0.5 / width, 0.5 / height)
    uvb = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x + hp.x, 0., 1. - hp.x) * repeat_h)
    uvb.y = (tm.clamp(uv.y + hp.y, 0., 1. - hp.y) * repeat_w)
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = tm.clamp(pos.y, 0., height - 1.)
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = tm.min(x0+1, int(width - 1))
    y1 = tm.min(y0+1, int(height - 1))
    return tm.ivec4(x0, y0, x1, y1), tm.vec2(dx, dy)

@ti.func
def _sample_bilinear_repeat_partial(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    uvb = (uv * tm.vec2(repeat_h, repeat_w)) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = pos.y % height
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = (x0+1) % width
    y1 = (y0+1) % height
    return tm.ivec4(x0, y0, x1, y1), tm.vec2(dx, dy)

@ti.func
def _sample_bilinear_repeat_x_partial(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    hpy = 0.5 / height
    uvb = tm.vec2(0.)
    uvb.x = (uv.x * repeat_h) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hpy, 1. - hpy) * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = tm.clamp(pos.x, 0., width - 1.)
    pos.y = pos.y % height
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = tm.min(x0+1, int(width - 1))
    y1 = (y0+1) % height
    return tm.ivec4(x0, y0, x1, y1), tm.vec2(dx, dy)

@ti.func
def _sample_bilinear_repeat_y_partial(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    hpx = 0.5 / width
    uvb = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hpx, 1. - hpx) * repeat_h) % 1.
    uvb.y = (uv.y * repeat_w) % 1.
    pos = tm.vec2(uvb.x * width - 0.5, uvb.y * height - 0.5)
    pos.x = pos.x % width
    pos.y = tm.clamp(pos.y, 0., height - 1.)
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = (x0+1) % width
    y1 = tm.min(y0+1, int(height - 1))
    return tm.ivec4(x0, y0, x1, y1), tm.vec2(dx, dy)

@ti.func
def sample_indexed_bilinear(
    tex:ti.template(), 
    uv:tm.vec2, 
    idx:int,
    wrap_mode:int, 
    repeat_w:int, 
    repeat_h:int, 
    window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)

    xy = tm.ivec4(0)
    dxdy = tm.vec2(0.)
    if wrap_mode == WrapMode.CLAMP:
        xy, dxdy = _sample_bilinear_clamp_partial(uv, width, height, repeat_w, repeat_h)
    elif wrap_mode == WrapMode.REPEAT:
        xy, dxdy = _sample_bilinear_repeat_partial(uv, width, height, repeat_w, repeat_h)
    elif wrap_mode == WrapMode.REPEAT_X:
        xy, dxdy = _sample_bilinear_repeat_x_partial(uv, width, height, repeat_w, repeat_h)
    elif wrap_mode == WrapMode.REPEAT_Y:
        xy, dxdy = _sample_bilinear_repeat_y_partial(uv, width, height, repeat_w, repeat_h)

    xofs, yofs = int(window.x), int(window.y)
    q00 = tex[idx, yofs + xy.y, xofs + xy.x] 
    q01 = tex[idx, yofs + xy.w, xofs + xy.x] 
    q10 = tex[idx, yofs + xy.y, xofs + xy.z] 
    q11 = tex[idx, yofs + xy.w, xofs + xy.z] 
    
    q0 = tm.mix(q00, q10, dxdy.x)
    q1 = tm.mix(q01, q11, dxdy.x)

    if ti.static(tex.n == 1):
        out = 0.
        out = tm.mix(q0, q1, dxdy.y)
        return out
    elif ti.static(tex.n == 2):
        out = tm.vec2(0.)
        out = tm.mix(q0, q1, dxdy.y)
        return out
    elif ti.static(tex.n == 3):
        out = tm.vec3(0.)
        out = tm.mix(q0, q1, dxdy.y)
        return out
    elif ti.static(tex.n == 4):
        out = tm.vec4(0.)
        out = tm.mix(q0, q1, dxdy.y)
        return out




@ti.data_oriented
class Sampler2D:
    """
    Taichi 2D texture sampler.

    :param repeat_w: Number of times to repeat image horizontally.
    :param repeat_h: Number of times to repeat image vertically.
    :param filter_mode: Filter mode.
    :param wrap_mode: Wrap mode.
    """
    def __init__(self, 
        repeat_w:int=1, 
        repeat_h:int=1, 
        filter_mode:Union[FilterMode, str]=FilterMode.BILINEAR, 
        wrap_mode:Union[WrapMode, str]=WrapMode.REPEAT
        ):
        self.repeat_w = repeat_w
        self.repeat_h = repeat_h
        self.filter_mode = int(filter_mode) if isinstance(filter_mode, FilterMode) else FilterMode[filter_mode.strip().upper()]
        self.wrap_mode = int(wrap_mode) if isinstance(wrap_mode, WrapMode) else WrapMode[wrap_mode.strip().upper()]

        # a few reasons to abort
        if not (self.filter_mode & FilterMode.SUPPORTED_2D):
            raise Exception("Unsupported Texture2D filter mode: " + self.filter_mode.name)
        if not (self.wrap_mode & WrapMode.SUPPORTED_2D):
            raise Exception("Unsupported Texture2D wrap mode: " + self.wrap_mode.name)

    @ti.func
    def _get_lod_window(self, tex:ti.template(), lod:float) -> tm.ivec4:
        ml = int(tm.min(lod, tex.max_mip))
        window = tm.ivec4(0)
        window.x = tex.width if ml > 0 else 0
        window.z = window.x + (tex.width >> ml)
        window.y = tex.height - (tex.height >> tm.max(ml - 1, 0))
        window.w = window.y + (tex.height >> ml)

        return window

    @ti.func
    def _get_lod_windows(self, tex:ti.template(), lod:float) -> (tm.ivec4, tm.ivec4):
        ml_high = int(tm.min(tm.ceil(lod), tex.max_mip))
        ml_low = int(tm.min(tm.floor(lod), tex.max_mip))
        window_high, window_low = tm.ivec4(0), tm.ivec4(0)
        
        window_low.x = tex.width if ml_low > 0 else 0
        window_high.x = tex.width

        window_low.z = window_low.x + (tex.width >> ml_low)
        window_high.z = window_high.x + (tex.width >> ml_high)

        window_low.y = tex.height - (tex.height >> tm.max(ml_low - 1, 0))
        window_high.y = tex.height - (tex.height >> (ml_high - 1))

        window_low.w = window_low.y + (tex.height >> ml_low)
        window_high.w = window_high.y + (tex.height >> ml_high)

        return window_high, window_low

    @ti.func
    def _sample_window(self, tex:ti.template(), uv:tm.vec2, window:tm.ivec4):
        if ti.static(tex.channels == 1):
            if ti.static(self.filter_mode == FilterMode.NEAREST):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_nn_r_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_nn_r_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_nn_r_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_nn_r_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.BILINEAR):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_bilinear_r_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_bilinear_r_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_bilinear_r_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_bilinear_r_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.B_SPLINE):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_b_spline_r_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_b_spline_r_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_b_spline_r_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_b_spline_r_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.BICUBIC):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_cubic_r_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_cubic_r_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_cubic_r_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_cubic_r_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.MITCHELL_NETRAVALI):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_r_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_r_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_r_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_r_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
            elif ti.static(self.filter_mode == FilterMode.CATMULL_ROM):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_r_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_r_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_r_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_r_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
        elif ti.static(tex.channels == 2):
            if ti.static(self.filter_mode == FilterMode.NEAREST):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_nn_rg_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window).rg
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_nn_rg_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window).rg
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_nn_rg_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window).rg
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_nn_rg_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window).rg
            elif ti.static(self.filter_mode == FilterMode.BILINEAR):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_bilinear_rg_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window).rg
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_bilinear_rg_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window).rg
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_bilinear_rg_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window).rg
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_bilinear_rg_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window).rg
            elif ti.static(self.filter_mode == FilterMode.B_SPLINE):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_b_spline_rg_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_b_spline_rg_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_b_spline_rg_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_b_spline_rg_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.BICUBIC):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_cubic_rg_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_cubic_rg_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_cubic_rg_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_cubic_rg_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.MITCHELL_NETRAVALI):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rg_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rg_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rg_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rg_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
            elif ti.static(self.filter_mode == FilterMode.CATMULL_ROM):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rg_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rg_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rg_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rg_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
        elif ti.static(tex.channels == 3):
            if ti.static(self.filter_mode == FilterMode.NEAREST):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_nn_rgb_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_nn_rgb_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_nn_rgb_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_nn_rgb_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window).rgb
            elif ti.static(self.filter_mode == FilterMode.BILINEAR):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_bilinear_rgb_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_bilinear_rgb_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_bilinear_rgb_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_bilinear_rgb_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window).rgb
            elif ti.static(self.filter_mode == FilterMode.B_SPLINE):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_b_spline_rgb_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_b_spline_rgb_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_b_spline_rgb_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_b_spline_rgb_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.BICUBIC):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_cubic_rgb_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_cubic_rgb_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_cubic_rgb_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_cubic_rgb_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.MITCHELL_NETRAVALI):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rgb_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rgb_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rgb_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rgb_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
            elif ti.static(self.filter_mode == FilterMode.CATMULL_ROM):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rgb_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rgb_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rgb_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rgb_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
        elif ti.static(tex.channels == 4):
            if ti.static(self.filter_mode == FilterMode.NEAREST):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_nn_rgba_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_nn_rgba_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_nn_rgba_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_nn_rgba_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window).rgba
            elif ti.static(self.filter_mode == FilterMode.BILINEAR):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_bilinear_rgba_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_bilinear_rgba_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_bilinear_rgba_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_bilinear_rgba_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window).rgba
            elif ti.static(self.filter_mode == FilterMode.B_SPLINE):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_b_spline_rgba_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_b_spline_rgba_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_b_spline_rgba_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_b_spline_rgba_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.BICUBIC):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_cubic_rgba_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_cubic_rgba_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_cubic_rgba_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_cubic_rgba_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.MITCHELL_NETRAVALI):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rgba_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rgba_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rgba_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rgba_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, 0.333333, 0.333333)
            elif ti.static(self.filter_mode == FilterMode.CATMULL_ROM):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rgba_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rgba_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rgba_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rgba_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, 0., 0.5)

    @ti.func
    def sample(self, tex:ti.template(), uv:tm.vec2):
        """
        Sample texture at uv coordinates.
        
        :param tex: Texture to sample.
        :type tex: Texture2D
        :param uv: UV coordinates.
        :type uv: taichi.math.vec2
        :return: Sampled texel, subject to filter mode interpolation.
        :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
        """
        window = tm.ivec4(0, 0, tex.width, tex.height)
        # "out" variable must be here or things break.
        # We can't return directly on Linux or taking e.g. `sample(...).rgb` will throw:
        # Assertion failure: var.cast<IndexExpression>()->is_matrix_field() || var.cast<IndexExpression>()->is_ndarray()
        # Your guess as to wtf that means is as good as mine.
        out = self._sample_window(tex, uv, window) 
        return out 

    @ti.func
    def sample_lod(self, tex:ti.template(), uv:tm.vec2, lod:float):
        """
        Sample texture at uv coordinates at specified mip level.

        :param tex: Texture to sample.
        :type tex: Texture2D
        :param uv: UV coordinates.
        :type uv: taichi.math.vec2
        :param lod: Level of detail.
        :type lod: float
        :return: Sampled texel, subject to filter mode interpolation.
        :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
        """
        if ti.static(tex.channels == 1):
            out = 0.
            if tex.max_mip == 0:
                out = self.sample(tex, uv)
            elif lod % 1. == 0.:
                window = self._get_lod_window(tex, lod)
                out = self._sample_window(tex, uv, window)
            else:
                window_high, window_low = self._get_lod_windows(tex, lod)
                low = self._sample_window(tex, uv, window_low)
                high = self._sample_window(tex, uv, window_high)
                out = tm.mix(low, high, lod % 1.)
            return out
        if ti.static(tex.channels == 2):
            out = tm.vec2(0.)
            if tex.max_mip == 0:
                out = self.sample(tex, uv)
            elif lod % 1. == 0.:
                window = self._get_lod_window(tex, lod)
                out = self._sample_window(tex, uv, window)
            else:
                window_high, window_low = self._get_lod_windows(tex, lod)
                low = self._sample_window(tex, uv, window_low)
                high = self._sample_window(tex, uv, window_high)
                out = tm.mix(low, high, lod % 1.)
            return out
        if ti.static(tex.channels == 3):
            out = tm.vec3(0.)
            if tex.max_mip == 0:
                out = self.sample(tex, uv)
            elif lod % 1. == 0.:
                window = self._get_lod_window(tex, lod)
                out = self._sample_window(tex, uv, window)
            else:
                window_high, window_low = self._get_lod_windows(tex, lod)
                low = self._sample_window(tex, uv, window_low)
                high = self._sample_window(tex, uv, window_high)
                out = tm.mix(low, high, lod % 1.)
            return out
        if ti.static(tex.channels == 4):
            out = tm.vec4(0.)
            if tex.max_mip == 0:
                out = self.sample(tex, uv)
            elif lod % 1. == 0.:
                window = self._get_lod_window(tex, lod)
                out = self._sample_window(tex, uv, window)
            else:
                window_high, window_low = self._get_lod_windows(tex, lod)
                low = self._sample_window(tex, uv, window_low)
                high = self._sample_window(tex, uv, window_high)
                out = tm.mix(low, high, lod % 1.)
            return out

    @ti.func
    def _fetch_r(self, tex:ti.template(), xy:tm.ivec2) -> float:
        return tex.field[xy.y, xy.x]

    @ti.func
    def _fetch_rg(self, tex:ti.template(), xy:tm.ivec2) -> tm.vec2:
        return tm.vec2(tex.field[xy.y, xy.x])

    @ti.func
    def _fetch_rgb(self, tex:ti.template(), xy:tm.ivec2) -> tm.vec3:
        return tm.vec3(tex.field[xy.y, xy.x])

    @ti.func
    def _fetch_rgba(self, tex:ti.template(), xy:tm.ivec2) -> tm.vec4:
        return tm.vec4(tex.field[xy.y, xy.x])

    @ti.func
    def fetch(self, tex:ti.template(), xy:tm.ivec2):
        """Fetch texel at indexed xy location.
        
        :param tex: Texture to sample.
        :type tex: Texture2D
        :param xy: xy index.
        :type xy: taichi.math.ivec2
        :return: Sampled texel.
        :rtype: float | taichi.math.vec2 | taichi.math.vec3 | taichi.math.vec4
        """
        if ti.static(tex.channels == 1):
            return self._fetch_r(tex, xy)
        if ti.static(tex.channels == 2):
            return self._fetch_rg(tex, xy)
        if ti.static(tex.channels == 3):
            return self._fetch_rgb(tex, xy)
        if ti.static(tex.channels == 4):
            return self._fetch_rgba(tex, xy)


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
            self.channels = _count_channels(im)

            # prepare data for a ti field
            if self.channels == 1:      im = _prep_r(im, self.flip_y)
            elif self.channels == 2:    im = _prep_rg(im, self.flip_y)
            elif self.channels == 3:    im = _prep_rgb(im, self.flip_y)
            elif self.channels == 4:    im = _prep_rgba(im, self.flip_y)
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
            self.__populate_prepared(im)

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
        assert self.channels == _count_channels(im), f"image tensor must have {self.channels} channels; got {_count_channels(im)}"
        if self.channels == 1:      
            im = _prep_r(im, self.flip_y)
        elif self.channels == 2:    
            im = _prep_rg(im, self.flip_y)
        elif self.channels == 3:    
            im = _prep_rgb(im, self.flip_y)
        elif self.channels == 4:    
            im = _prep_rgba(im, self.flip_y)
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
        self.__populate_prepared(im.float()) 

    def __populate_prepared(self, im:torch.Tensor):
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

# This "_prep_x" stuff is ugly and redundant, but this is easier with some special handling.
#
# Valid input data here can be any of: 
# - [C, H, W] torch tensor
# - [H, W] torch tensor if C=1
# - [C] torch tensor (single color value)
# - [H, W, C] numpy array
# - [H, W] numpy array if C=1
# - [C] numpy array (single color value)
# - float, tm.vec2, tm.vec3, or tm.vec4 value
# - other numeric value if C=1
def _prep_r(val:torch.Tensor, flip_y:bool = False):
    """Converts image data to [H, W] floating point torch image tensor"""
    if torch.is_tensor(val) and val.dim() == 3:
        if val.size(0) != 1: val = val[0:1]
        val = val.permute(1, 2, 0)
    elif torch.is_tensor(val) and val.dim() == 2:
        val = val.unsqueeze(-1)
    elif torch.is_tensor(val) and val.dim() == 1 and val.size(0) == 1:
        val = val.unsqueeze(0)
    elif isinstance(val, np.ndarray) and len(val.shape) == 3 and val.shape[2] == 1:
        val = torch.from_numpy(val.squeeze(-1))
    elif isinstance(val, np.ndarray) and len(val.shape) == 2 and val.shape[2] == 1:
        val = torch.from_numpy(val)
    elif type(val) == int or type(val) == float or (type(val) == str and isnumber(val)):
        val = torch.tensor([[float(val)]], dtype=torch.float32)
    else: 
        raise Exception("Expected [C=1, H, W] image tensor, [H, W, C=1] ndarray or float value")
    if flip_y: val = torch.flip(val, dims=[0])
    return val.float()

def _prep_rg(val:torch.Tensor, flip_y:bool = False):
    """Converts image data to [H, W, C=2] floating point torch image tensor"""
    if torch.is_tensor(val) and val.dim() == 3 and val.size(0) == 2:
        val = val.permute(1, 2, 0) # C, H, W -> H, W, C
    elif torch.is_tensor(val) and val.dim() == 1 and val.size(0) == 2:
        val = val.unsqueeze(0).unsqueeze(0)
    elif isinstance(val, np.ndarray) and len(val.shape) == 3 and val.shape[2] == 2:
        val = torch.from_numpy(val)
    elif isinstance(val, np.ndarray) and len(val.shape) == 1 and val.shape[0] == 2:
        val = torch.from_numpy(val).unsqueeze(0).unsqueeze(0)
    elif (type(val) == ti.lang.matrix.Vector or type(val) == ti.lang.matrix.Matrix) and len(val) == 2:
        val = torch.tensor([[[val.r, val.g, val.b]]], dtype=torch.float32)
    else: raise Exception("Expected [C=2, H, W] image tensor, [H, W, C=2] ndarray or vec2 value")
    if flip_y: val = torch.flip(val, dims=[0])
    return val.float()

def _prep_rgb(val:torch.Tensor, flip_y:bool = False):
    """Converts image data to [H, W, C=3] floating point torch image tensor"""
    if torch.is_tensor(val) and val.dim() == 3 and val.size(0) == 3:
        val = val.permute(1, 2, 0) # C, H, W -> H, W, C
    elif torch.is_tensor(val) and val.dim() == 1 and val.size(0) == 3:
        val = val.unsqueeze(0).unsqueeze(0)
    elif isinstance(val, np.ndarray) and len(val.shape) == 3 and val.shape[2] == 3:
        val = torch.from_numpy(val)
    elif isinstance(val, np.ndarray) and len(val.shape) == 1 and val.shape[0] == 3:
        val = torch.from_numpy(val).unsqueeze(0).unsqueeze(0)
    elif (type(val) == ti.lang.matrix.Vector or type(val) == ti.lang.matrix.Matrix) and len(val) == 3:
        val = torch.tensor([[[val.r, val.g, val.b]]], dtype=torch.float32)
    else: raise Exception("Expected [C=3, H, W] image tensor, [H, W, C=3] ndarray or vec3 value")
    if flip_y: val = torch.flip(val, dims=[0])
    return val.float()

def _prep_rgba(val:torch.Tensor, flip_y:bool = False):
    """Converts image data to [H, W, C=4] floating point torch image tensor"""
    if torch.is_tensor(val) and val.dim() == 3 and val.size(0) == 4:
        val = val.permute(1, 2, 0) # C, H, W -> H, W, C
    elif torch.is_tensor(val) and val.dim() == 1 and val.size(0) == 4:
        val = val.unsqueeze(0).unsqueeze(0)
    elif isinstance(val, np.ndarray) and len(val.shape) == 3 and val.shape[2] == 4:
        val = torch.from_numpy(val)
    elif isinstance(val, np.ndarray) and len(val.shape) == 1 and val.shape[0] == 4:
        val = torch.from_numpy(val).unsqueeze(0).unsqueeze(0)
    elif (type(val) == ti.lang.matrix.Vector or type(val) == ti.lang.matrix.Matrix) and len(val) == 4:
        val = torch.tensor([[[val.r, val.g, val.b]]], dtype=torch.float32)
    else: raise Exception("Expected [C=4, H, W] image tensor, [H, W, C=4] ndarray or vec4 value")
    if flip_y: val = torch.flip(val, dims=[0])
    return val.float()

def _count_channels(im:torch.Tensor):
    channels = 0
    if   isinstance(im, float): channels = 1
    elif isinstance(im, torch.Tensor) and torch.is_floating_point(im):
        if   im.dim() == 1 and im.size(0) == 1: channels = 1
        elif im.dim() == 2 or (im.dim() == 3 and im.size(0) == 1): channels = 1
        elif (im.dim() == 3 or im.dim() == 1) and im.size(0) == 2: channels = 2
        elif (im.dim() == 3 or im.dim() == 1) and im.size(0) == 3: channels = 3
        elif (im.dim() == 3 or im.dim() == 1) and im.size(0) == 4: channels = 4
    elif isinstance(im, np.ndarray) and isinstance(im, np.floating):
        if   len(im.shape) == 1 and im.shape[0] == 1: channels = 1
        elif len(im.shape) == 2 or (len(im.shape) == 3 and im.shape[0] == 1): channels = 1
        elif (len(im.shape) == 3 or len(im.shape) == 1) and im.shape[0] == 2: channels = 2
        elif (len(im.shape) == 3 or len(im.shape) == 1) and im.shape[0] == 3: channels = 3
        elif (len(im.shape) == 3 or len(im.shape) == 0) and im.shape[0] == 4: channels = 4
    elif type(im) == ti.lang.matrix.Vector or type(im) == ti.lang.matrix.Matrix: 
        if   len(im) == 1: channels = 1
        elif len(im) == 2: channels = 2
        elif len(im) == 3: channels = 3
        elif len(im) == 4: channels = 4
    else: channels = 0
    return channels