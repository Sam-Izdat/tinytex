"""
texture
=================================
Taichi backend-independent texture sampling module
 """

import taichi as ti
import taichi.math as tm

import torch
import numpy as np
import typing
from typing import Union
from enum import IntEnum    

class FilterMode(IntEnum):
    NEAREST     = 1<<0
    BILINEAR    = 1<<1
    TRILINEAR   = 1<<2
    BICUBIC     = 1<<3

    SUPPORTED_2D = NEAREST | BILINEAR | BICUBIC
    SUPPORTED_3D = NEAREST | TRILINEAR

class WrapMode(IntEnum):
    REPEAT      = 1<<0
    CLAMP       = 1<<1
    REPEAT_X    = 1<<2
    REPEAT_Y    = 1<<3

    # TODO: MIRROR, etc

    SUPPORTED_2D = REPEAT | CLAMP | REPEAT_X | REPEAT_Y
    SUPPORTED_3D = REPEAT | CLAMP | REPEAT_X | REPEAT_Y

@ti.func
def sample_nn_repeat(
    tex:ti.template(), 
    uv:tm.vec2, 
    wrap_mode:int, 
    tile_x:int, 
    tile_y:int,  
    window:tm.ivec4) -> ti.template():
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = uv
    x, y = 0, 0
    uvb.x = uv.x % 1.
    uvb.y = uv.y % 1.
    x = int(window.x + ((uvb.x * float(width * tile_x)) % width))
    y = int(window.y + ((uvb.y * float(height * tile_y)) % height))    
    return tex[y, x]

@ti.func
def sample_nn_clamp(
    tex:ti.template(), 
    uv:tm.vec2, 
    wrap_mode:int, 
    tile_x:int, 
    tile_y:int,  
    window:tm.ivec4) -> ti.template():
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = uv
    x, y = 0, 0
    uvb.x = tm.clamp(uv.x, 0., 1. - (0.5 / width)) 
    uvb.y = tm.clamp(uv.y, 0., 1. - (0.5 / height))
    x = int(window.x + ((uvb.x * float(width * tile_x)) % width))
    y = int(window.y + ((uvb.y * float(height * tile_y)) % height))    
    return tex[y, x]

@ti.func
def sample_nn_repeat_x(
    tex:ti.template(), 
    uv:tm.vec2, 
    wrap_mode:int, 
    tile_x:int, 
    tile_y:int,  
    window:tm.ivec4) -> ti.template():
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = uv
    x, y = 0, 0
    uvb.x = uv.x % 1.
    uvb.y = tm.clamp(uv.y, 0., 1. - (0.5 / height))
    x = int(window.x + ((uvb.x * float(width * tile_x)) % width))
    y = int(window.y + ((uvb.y * float(height * tile_y)) % height))    
    return tex[y, x]

@ti.func
def sample_nn_repeat_y(
    tex:ti.template(), 
    uv:tm.vec2, 
    wrap_mode:int, 
    tile_x:int, 
    tile_y:int,  
    window:tm.ivec4) -> ti.template():
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = uv
    x, y = 0, 0
    uvb.x = tm.clamp(uv.x, 0., 1. - (0.5 / width)) 
    uvb.y = uv.y % 1.
    x = int(window.x + ((uvb.x * float(width * tile_x)) % width))
    y = int(window.y + ((uvb.y * float(height * tile_y)) % height))    
    return tex[y, x]


# previously - ti.real_func
@ti.func
def sample_nn_r_repeat(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> float:
    return sample_nn_repeat(tex, uv, tile_x, tile_y, window)

# previously - ti.real_func
@ti.func
def sample_nn_r_clamp(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> float:
    return sample_nn_clamp(tex, uv, tile_x, tile_y, window)

# previously - ti.real_func
@ti.func
def sample_nn_r_repeat_x(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> float:
    return sample_nn_repeat_x(tex, uv, tile_x, tile_y, window)

# previously - ti.real_func
@ti.func
def sample_nn_r_repeat_y(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> float:
    return sample_nn_repeat_y(tex, uv, tile_x, tile_y, window)



# previously - ti.real_func
@ti.func
def sample_nn_rg_repeat(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_nn_repeat(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_nn_rg_clamp(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_nn_clamp(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_nn_rg_repeat_x(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_nn_repeat_x(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_nn_rg_repeat_y(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_nn_repeat_y(tex, uv, tile_x, tile_y, window))



# previously - ti.real_func
@ti.func
def sample_nn_rgb_repeat(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_nn_repeat(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgb_clamp(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_nn_clamp(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgb_repeat_x(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_nn_repeat_x(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgb_repeat_y(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_nn_repeat_y(tex, uv, tile_x, tile_y, window))


# previously - ti.real_func
@ti.func
def sample_nn_rgba_repeat(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_nn_repeat(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgba_clamp(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_nn_clamp(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgba_repeat_x(tex:ti.template(),uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_nn_repeat_x(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_nn_rgba_repeat_y(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_nn_repeat_y(tex, uv, tile_x, tile_y, window))


@ti.func
def sample_bilinear_repeat(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * tile_y) % 1.
    uvb.y = (uv.y * tile_x) % 1.
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
def sample_bilinear_clamp(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hp.x, 1. - hp.x) * tile_y) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hp.y, 1. - hp.y) * tile_x) % 1.
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
def sample_bilinear_repeat_x(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hpy = 0.5 / height
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * tile_y) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hpy, 1. - hpy) * tile_x) % 1.
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
def sample_bilinear_repeat_y(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hpx = 0.5 / width
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hpx, 1. - hpx) * tile_y) % 1.
    uvb.y = (uv.y * tile_x) % 1.
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
def sample_bilinear_r_repeat(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> float:
    return sample_bilinear_repeat(tex, uv, tile_x, tile_y, window)

# previously - ti.real_func
@ti.func
def sample_bilinear_r_clamp(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> float:
    return sample_bilinear_clamp(tex, uv, tile_x, tile_y, window)

# previously - ti.real_func
@ti.func
def sample_bilinear_r_repeat_x(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> float:
    return sample_bilinear_repeat_x(tex, uv, tile_x, tile_y, window)

# previously - ti.real_func
@ti.func
def sample_bilinear_r_repeat_y(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> float:
    return sample_bilinear_repeat_y(tex, uv, tile_x, tile_y, window)


# previously - ti.real_func
@ti.func
def sample_bilinear_rg_repeat(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_bilinear_repeat(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rg_clamp(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_bilinear_clamp(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rg_repeat_x(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_bilinear_repeat_x(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rg_repeat_y(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_bilinear_repeat_y(tex, uv, tile_x, tile_y, window))


# previously - ti.real_func
@ti.func
def sample_bilinear_rgb_repeat(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_bilinear_repeat(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgb_clamp(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_bilinear_clamp(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgb_repeat_x(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_bilinear_repeat_x(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgb_repeat_y(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_bilinear_repeat_y(tex, uv, tile_x, tile_y, window))


# previously - ti.real_func
@ti.func
def sample_bilinear_rgba_repeat(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_bilinear_repeat(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgba_clamp(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_bilinear_clamp(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgba_repeat_x(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_bilinear_repeat_x(tex, uv, tile_x, tile_y, window))

# previously - ti.real_func
@ti.func
def sample_bilinear_rgba_repeat_y(tex:ti.template(), uv:tm.vec2, tile_x:int, tile_y:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_bilinear_repeat_y(tex, uv, tile_x, tile_y, window))




@ti.func
def _sample_bilinear_clamp_partial(
    uv:tm.vec2,
    width:int,
    height:int,
    tile_x:int,
    tile_y:int,
    ) -> tuple:
    hp = tm.vec2(0.5 / width, 0.5 / height)
    uvb = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x + hp.x, 0., 1. - hp.x) * tile_y)
    uvb.y = (tm.clamp(uv.y + hp.y, 0., 1. - hp.y) * tile_x)
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
    tile_x:int,
    tile_y:int,
    ) -> tuple:
    uvb = (uv * tm.vec2(tile_y, tile_x)) % 1.
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
    tile_x:int,
    tile_y:int,
    ) -> tuple:
    hpy = 0.5 / height
    uvb = tm.vec2(0.)
    uvb.x = (uv.x * tile_y) % 1.
    uvb.y = (tm.clamp(uv.y, 0. + hpy, 1. - hpy) * tile_x) % 1.
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
    tile_x:int,
    tile_y:int,
    ) -> tuple:
    hpx = 0.5 / width
    uvb = tm.vec2(0.)
    uvb.x = (tm.clamp(uv.x, 0. + hpx, 1. - hpx) * tile_y) % 1.
    uvb.y = (uv.y * tile_x) % 1.
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
    tile_x:int, 
    tile_y:int, 
    window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)

    xy = tm.ivec4(0)
    dxdy = tm.vec2(0.)
    if wrap_mode == WrapMode.CLAMP:
        xy, dxdy = _sample_bilinear_clamp_partial(uv, width, height, tile_x, tile_y)
    elif wrap_mode == WrapMode.REPEAT:
        xy, dxdy = _sample_bilinear_repeat_partial(uv, width, height, tile_x, tile_y)
    elif wrap_mode == WrapMode.REPEAT_X:
        xy, dxdy = _sample_bilinear_repeat_x_partial(uv, width, height, tile_x, tile_y)
    elif wrap_mode == WrapMode.REPEAT_Y:
        xy, dxdy = _sample_bilinear_repeat_y_partial(uv, width, height, tile_x, tile_y)

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
class TiSampler2D:
    def __init__(self, 
        tile_x:int=1, 
        tile_y:int=1, 
        filter_mode:Union[FilterMode, str]=FilterMode.BILINEAR, 
        wrap_mode:Union[WrapMode, str]=WrapMode.REPEAT
        ):
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.filter_mode = int(filter_mode) if isinstance(filter_mode, FilterMode) else FilterMode[filter_mode.strip().upper()]
        self.wrap_mode = int(wrap_mode) if isinstance(wrap_mode, WrapMode) else WrapMode[wrap_mode.strip().upper()]

        # a few reasons to abort
        if not (self.filter_mode & FilterMode.SUPPORTED_2D):
            raise Exception("Unsupported TiTexture2D filter mode: " + self.filter_mode.name)
        if not (self.wrap_mode & WrapMode.SUPPORTED_2D):
            raise Exception("Unsupported TiTexture2D wrap mode: " + self.wrap_mode.name)

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
                    return sample_nn_r_repeat(tex.field, uv, self.tile_x, self.tile_y, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_nn_r_clamp(tex.field, uv, self.tile_x, self.tile_y, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_nn_r_repeat_x(tex.field, uv, self.tile_x, self.tile_y, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_nn_r_repeat_y(tex.field, uv, self.tile_x, self.tile_y, window)
            elif ti.static(self.filter_mode == FilterMode.BILINEAR):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_bilinear_r_repeat(tex.field, uv, self.tile_x, self.tile_y, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_bilinear_r_clamp(tex.field, uv, self.tile_x, self.tile_y, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_bilinear_r_repeat_x(tex.field, uv, self.tile_x, self.tile_y, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_bilinear_r_repeat_y(tex.field, uv, self.tile_x, self.tile_y, window)
        elif ti.static(tex.channels == 2):
            if ti.static(self.filter_mode == FilterMode.NEAREST):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_nn_rg_repeat(tex.field, uv, self.tile_x, self.tile_y, window).rg
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_nn_rg_clamp(tex.field, uv, self.tile_x, self.tile_y, window).rg
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_nn_rg_repeat_x(tex.field, uv, self.tile_x, self.tile_y, window).rg
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_nn_rg_repeat_y(tex.field, uv, self.tile_x, self.tile_y, window).rg
            elif ti.static(self.filter_mode == FilterMode.BILINEAR):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_bilinear_rg_repeat(tex.field, uv, self.tile_x, self.tile_y, window).rg
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_bilinear_rg_clamp(tex.field, uv, self.tile_x, self.tile_y, window).rg
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_bilinear_rg_repeat_x(tex.field, uv, self.tile_x, self.tile_y, window).rg
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_bilinear_rg_repeat_y(tex.field, uv, self.tile_x, self.tile_y, window).rg
        elif ti.static(tex.channels == 3):
            if ti.static(self.filter_mode == FilterMode.NEAREST):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_nn_rgb_repeat(tex.field, uv, self.tile_x, self.tile_y, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_nn_rgb_clamp(tex.field, uv, self.tile_x, self.tile_y, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_nn_rgb_repeat_x(tex.field, uv, self.tile_x, self.tile_y, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_nn_rgb_repeat_y(tex.field, uv, self.tile_x, self.tile_y, window).rgb
            elif ti.static(self.filter_mode == FilterMode.BILINEAR):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_bilinear_rgb_repeat(tex.field, uv, self.tile_x, self.tile_y, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_bilinear_rgb_clamp(tex.field, uv, self.tile_x, self.tile_y, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_bilinear_rgb_repeat_x(tex.field, uv, self.tile_x, self.tile_y, window).rgb
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_bilinear_rgb_repeat_y(tex.field, uv, self.tile_x, self.tile_y, window).rgb
        elif ti.static(tex.channels == 4):
            if ti.static(self.filter_mode == FilterMode.NEAREST):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_nn_rgba_repeat(tex.field, uv, self.tile_x, self.tile_y, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_nn_rgba_clamp(tex.field, uv, self.tile_x, self.tile_y, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_nn_rgba_repeat_x(tex.field, uv, self.tile_x, self.tile_y, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_nn_rgba_repeat_y(tex.field, uv, self.tile_x, self.tile_y, window).rgba
            elif ti.static(self.filter_mode == FilterMode.BILINEAR):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_bilinear_rgba_repeat(tex.field, uv, self.tile_x, self.tile_y, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_bilinear_rgba_clamp(tex.field, uv, self.tile_x, self.tile_y, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_bilinear_rgba_repeat_x(tex.field, uv, self.tile_x, self.tile_y, window).rgba
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_bilinear_rgba_repeat_y(tex.field, uv, self.tile_x, self.tile_y, window).rgba

    @ti.func
    def sample(self, tex:ti.template(), uv:tm.vec2):
        window = tm.ivec4(0, 0, tex.width, tex.height)
        # "out" variable must be here or things break.
        # We can't return directly on Linux or taking e.g. `sample(...).rgb` will throw:
        # Assertion failure: var.cast<IndexExpression>()->is_matrix_field() || var.cast<IndexExpression>()->is_ndarray()
        # Your guess as to wtf that means is as good as mine.
        out = self._sample_window(tex, uv, window) 
        return out 

    @ti.func
    def sample_lod(self, tex:ti.template(), uv:tm.vec2, lod:float):
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
        if ti.static(tex.channels == 1):
            return self._fetch_r(tex, xy)
        if ti.static(tex.channels == 2):
            return self._fetch_rg(tex, xy)
        if ti.static(tex.channels == 3):
            return self._fetch_rgb(tex, xy)
        if ti.static(tex.channels == 4):
            return self._fetch_rgba(tex, xy)





























@ti.data_oriented
class TiTexture2D:
    def __init__(self, 
        im:Union[torch.Tensor, tuple],
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
                HO = 0
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

    # def __del__(self):
    #     if self.fb_snode_tree: 
    #         self.fb_snode_tree.destroy()
            
    def populate(self, im:torch.Tensor):
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
            HO = 0
            tmp = torch.zeros(C, H, W+W//2)
            tmp[:, 0:H, 0:W] = im
            self.max_mip = H.bit_length() - 1

            # To generate with torch:
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
        """Regenerate texture mip chain from level 0 and populate Taichi field."""
        window_last = tm.ivec4(0, 0, self.width, self.height)
        window = tm.ivec4(0)
        for _ in range(1):
            for ml in range(1, self.max_mip + 1):
                window.x = self.width if ml > 0 else 0
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