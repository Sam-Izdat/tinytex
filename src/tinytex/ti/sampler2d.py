"""
sampler2d
=================================
Taichi 2D texture sampling module. Supports CPU, CUDA and Vulkan backends.
 """

# NOTE: The filtering here is inefficient and does not even use hardware-native sampling. 
# The purpose of this module is convenience, but there's a lot of room for optimization. See: 
# https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering
# Radiance cascades probe grids, however, cannot have adjacent texels and so, as hardware linear interpolation can't be exploited,
# they are not suitable for third order interpolation/approximation with the above approach.

import typing
from typing import Union

import taichi as ti
import taichi.math as tm

import torch
import numpy as np    

from .params import *
from .splines import *


@ti.data_oriented
class Sampler2D:
    """
    Taichi 2D texture sampler.

    :param repeat_w: Number of times to repeat image width/x.
    :param repeat_h: Number of times to repeat image height/y.
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
            elif ti.static(self.filter_mode == FilterMode.HERMITE):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_hermite_r_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_hermite_r_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_hermite_r_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_hermite_r_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.MITCHELL_NETRAVALI):
                third = 0.3333333
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_r_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_r_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_r_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_r_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
            elif ti.static(self.filter_mode == FilterMode.CATMULL_ROM):
                b, c = 0., 0.5
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_r_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_r_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_r_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_r_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
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
            elif ti.static(self.filter_mode == FilterMode.HERMITE):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_hermite_rg_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_hermite_rg_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_hermite_rg_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_hermite_rg_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.MITCHELL_NETRAVALI):
                third = 0.3333333
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rg_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rg_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rg_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rg_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
            elif ti.static(self.filter_mode == FilterMode.CATMULL_ROM):
                b, c = 0., 0.5
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rg_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rg_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rg_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rg_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
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
            elif ti.static(self.filter_mode == FilterMode.HERMITE):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_hermite_rgb_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_hermite_rgb_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_hermite_rgb_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_hermite_rgb_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.MITCHELL_NETRAVALI):
                third = 0.3333333
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rgb_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rgb_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rgb_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rgb_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
            elif ti.static(self.filter_mode == FilterMode.CATMULL_ROM):
                b, c = 0., 0.5
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rgb_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rgb_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rgb_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rgb_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
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
            elif ti.static(self.filter_mode == FilterMode.HERMITE):
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_hermite_rgba_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_hermite_rgba_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_hermite_rgba_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_hermite_rgba_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window)
            elif ti.static(self.filter_mode == FilterMode.MITCHELL_NETRAVALI):
                third = 0.3333333
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rgba_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rgba_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rgba_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rgba_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, third, third)
            elif ti.static(self.filter_mode == FilterMode.CATMULL_ROM):
                b, c = 0., 0.5
                if ti.static(self.wrap_mode == WrapMode.REPEAT):
                    return sample_mitchell_netravali_rgba_repeat(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.CLAMP):
                    return sample_mitchell_netravali_rgba_clamp(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_X):
                    return sample_mitchell_netravali_rgba_repeat_x(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)
                elif ti.static(self.wrap_mode == WrapMode.REPEAT_Y):
                    return sample_mitchell_netravali_rgba_repeat_y(tex.field, uv, self.repeat_w, self.repeat_h, window, b, c)

    @ti.func
    def sample(self, tex:ti.template(), uv:tm.vec2):
        """
        Sample texture at uv coordinates.
        
        :param tex: Texture to sample.
        :type tex: Texture2D
        :param uv: UV coordinates.
        :type uv: taichi.math.vec2
        :return: Filtered sampled texel.
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
        :return: Filtered sampled texel.
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

# ------------------------------------------------------

@ti.func
def dxdy_bilinear_clamp(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.) 
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = tm.min(x0+1, int(width - 1))
    y1 = tm.min(y0+1, int(height - 1))
    return tm.ivec4(x0, y0, x1, y1), tm.vec2(dx, dy)

@ti.func
def dxdy_bilinear_repeat(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    uvb = (uv * tm.vec2(repeat_w, repeat_h)) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = (pos.y - 0.5) % height
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = (x0+1) % width
    y1 = (y0+1) % height
    return tm.ivec4(x0, y0, x1, y1), tm.vec2(dx, dy)

@ti.func
def dxdy_bilinear_repeat_x(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    hpy = 0.5 / height
    eps = 1e-7
    uvb = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hpy, 1. - hpy)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = (x0+1) % width
    y1 = tm.min(y0+1, int(height - 1))
    return tm.ivec4(x0, y0, x1, y1), tm.vec2(dx, dy)

@ti.func
def dxdy_bilinear_repeat_y(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    hpx = 0.5 / width
    eps = 1e-7
    uvb = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hpx, 1. - hpx)
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = (pos.y - 0.5) % height
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = tm.min(x0+1, int(width - 1))
    y1 = (y0+1) % height
    return tm.ivec4(x0, y0, x1, y1), tm.vec2(dx, dy)


@ti.func
def dxdy_2D_scoped_grid_bilinear(
    xy_kernel:tm.vec2, 
    grid_height:int, 
    grid_width:int, 
    probe_height:int, 
    probe_width:int,
    probe_pad:int,
    kernel_height:float,
    kernel_width:float,
    wrap_mode:int) -> (tm.ivec4, tm.vec2):
    """
    Return bilinear indices and deltas for 2D grid, subdividing a kernel, given denormalized xy texel position.
    Intended for probe grid interpolation. Kernel size excludes padded grid cells. Example:

    .. highlight:: text
    .. code-block:: text
        
        xy_kernel:     [5, 3]
        grid_height:   3
        grid_width:    4
        probe_height:  2
        probe_width:   3
        probe_pad:     0
        kernel_height: 6
        kernel_width:  12

        |ooo|ooo|ooo|ooo|
        |ooo|ooo|ooo|ooo|
        -----------------
        |ooo|oox|ooo|ooo|
        |ooo|ooo|ooo|ooo|
        -----------------
        |ooo|ooo|ooo|ooo|
        |ooo|ooo|ooo|ooo|
    """
    padded_kernel = tm.vec2(kernel_width + probe_pad * probe_width * 2, kernel_height + probe_pad * probe_height * 2)
    xy = tm.vec2(xy_kernel.x + probe_pad * probe_width, xy_kernel.y + probe_pad * probe_height)
    uv_grid = xy / padded_kernel

    indices, dxdy = tm.ivec4(0), tm.vec2(0.)
    if wrap_mode == WrapMode.CLAMP:
        indices, dxdy = dxdy_bilinear_clamp(uv_grid, grid_width, grid_height, 1, 1)
    elif wrap_mode == WrapMode.REPEAT:
        indices, dxdy = dxdy_bilinear_repeat(uv_grid, grid_width, grid_height, 1, 1)
    elif wrap_mode == WrapMode.REPEAT_X:
        indices, dxdy = dxdy_bilinear_repeat_x(uv_grid, grid_width, grid_height, 1, 1)
    elif wrap_mode == WrapMode.REPEAT_Y:
        indices, dxdy = dxdy_bilinear_repeat_y(uv_grid, grid_width, grid_height, 1, 1)

    # Fetch and linearize quadrant indices
    indices = tm.ivec4(
        int((indices.y * grid_width) + indices.x), # q00
        int((indices.w * grid_width) + indices.x), # q01
        int((indices.y * grid_width) + indices.z), # q10
        int((indices.w * grid_width) + indices.z)  # q11
        )
    return indices, dxdy



@ti.func
def dxdy_cubic_clamp(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.) 
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0, y0 = tm.clamp(x1-1, 0, width-1), tm.clamp(y1-1, 0, height-1)
    x2, y2 = tm.clamp(x1+1, 0, width-1), tm.clamp(y1+1, 0, height-1)
    x3, y3 = tm.clamp(x1+2, 0, width-1), tm.clamp(y1+2, 0, height-1)
    dx, dy = pos.x - x1, pos.y - y1

    x_indices = tm.ivec4(x0, x1, x2, x3)
    y_indices = tm.ivec4(y0, y1, y2, y3)

    return x_indices, y_indices, tm.vec2(dx, dy)

@ti.func
def dxdy_cubic_repeat(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    uvb = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = (pos.y - 0.5) % height

    x1, y1 = int(pos.x), int(pos.y)
    x0, y0 = (x1-1) % width, (y1-1) % height
    x2, y2 = (x1+1) % width, (y1+1) % height
    x3, y3 = (x1+2) % width, (y1+2) % height
    dx, dy = pos.x - x1, pos.y - y1

    x_indices = tm.ivec4(x0, x1, x2, x3)
    y_indices = tm.ivec4(y0, y1, y2, y3)

    return x_indices, y_indices, tm.vec2(dx, dy)

@ti.func
def dxdy_cubic_repeat_x(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0, y0 = (x1-1) % width, tm.clamp(y1-1, 0, height-1)
    x2, y2 = (x1+1) % width, tm.clamp(y1+1, 0, height-1)
    x3, y3 = (x1+2) % width, tm.clamp(y1+2, 0, height-1)
    dx, dy = pos.x - x1, pos.y - y1

    x_indices = tm.ivec4(x0, x1, x2, x3)
    y_indices = tm.ivec4(y0, y1, y2, y3)

    return x_indices, y_indices, tm.vec2(dx, dy)

@ti.func
def dxdy_cubic_repeat_y(
    uv:tm.vec2,
    width:int,
    height:int,
    repeat_w:int,
    repeat_h:int,
    ) -> tuple:
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.) 
    pos.y = (pos.y - 0.5) % height

    x1, y1 = int(pos.x), int(pos.y)
    x0, y0 = tm.clamp(x1-1, 0, width-1), (y1-1) % height
    x2, y2 = tm.clamp(x1+1, 0, width-1), (y1+1) % height
    x3, y3 = tm.clamp(x1+2, 0, width-1), (y1+2) % height
    dx, dy = pos.x - x1, pos.y - y1

    x_indices = tm.ivec4(x0, x1, x2, x3)
    y_indices = tm.ivec4(y0, y1, y2, y3)

    return x_indices, y_indices, tm.vec2(dx, dy)

@ti.func
def dxdy_2D_scoped_grid_cubic(
    xy_kernel:tm.vec2, 
    grid_height:int, 
    grid_width:int, 
    probe_height:int, 
    probe_width:int,
    probe_pad:int,
    kernel_height:float,
    kernel_width:float,
    wrap_mode:int) -> (tm.mat4, tm.vec2): #(ivec16, vec16, tm.vec2):
    """
    Return cubic indices and deltas for 2D grid, subdividing a kernel, given denormalized xy texel position.
    Intended for probe grid intepolation/approximation. Kernel size excludes padded grid cells. Example:

    .. highlight:: text
    .. code-block:: text
        
        xy_kernel:     [5, 3]
        grid_height:   3
        grid_width:    4
        probe_height:  2
        probe_width:   3
        probe_pad:     0
        kernel_height: 6
        kernel_width:  12

        |ooo|ooo|ooo|ooo|
        |ooo|ooo|ooo|ooo|
        -----------------
        |ooo|oox|ooo|ooo|
        |ooo|ooo|ooo|ooo|
        -----------------
        |ooo|ooo|ooo|ooo|
        |ooo|ooo|ooo|ooo|
    """

    padded_kernel = tm.vec2(kernel_width + probe_pad * probe_width * 2, kernel_height + probe_pad * probe_height * 2)
    xy = tm.vec2(xy_kernel.x + probe_pad * probe_width, xy_kernel.y + probe_pad * probe_height)
    uv_grid = xy / padded_kernel

    ix, iy, dxdy = tm.ivec4(0), tm.ivec4(0), tm.vec2(0.)
    if wrap_mode == WrapMode.CLAMP:
        ix, iy, dxdy = dxdy_cubic_clamp(uv_grid, grid_width, grid_height, 1, 1)
    elif wrap_mode == WrapMode.REPEAT:
        ix, iy, dxdy = dxdy_cubic_repeat(uv_grid, grid_width, grid_height, 1, 1)
    elif wrap_mode == WrapMode.REPEAT_X:
        ix, iy, dxdy = dxdy_cubic_repeat_x(uv_grid, grid_width, grid_height, 1, 1)
    elif wrap_mode == WrapMode.REPEAT_Y:
        ix, iy, dxdy = dxdy_cubic_repeat_y(uv_grid, grid_width, grid_height, 1, 1)

    gw = grid_width
    indices = tm.mat4([
        [(iy[0] * gw) + ix[0], (iy[1] * gw) + ix[0], (iy[2] * gw) + ix[0], (iy[3] * gw) + ix[0]],
        [(iy[0] * gw) + ix[1], (iy[1] * gw) + ix[1], (iy[2] * gw) + ix[1], (iy[3] * gw) + ix[1]],
        [(iy[0] * gw) + ix[2], (iy[1] * gw) + ix[2], (iy[2] * gw) + ix[2], (iy[3] * gw) + ix[2]],
        [(iy[0] * gw) + ix[3], (iy[1] * gw) + ix[3], (iy[2] * gw) + ix[3], (iy[3] * gw) + ix[3]]
        ])

    return indices, dxdy

# ------------------------------------------------------

@ti.func
def sample_hermite_repeat_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = (pos.y - 0.5) % height

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
    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(filter_cubic_hermite(p, dx, dy), 0.)

    return out

@ti.func
def sample_hermite_repeat_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = (pos.y - 0.5) % height

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
    out = tm.max(filter_cubic_hermite(p, dx, dy), 0.)

    return out

@ti.func
def sample_hermite_clamp_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

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

    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(filter_cubic_hermite(p, dx, dy), 0.)

    return out

@ti.func
def sample_hermite_clamp_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

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
    out = tm.max(filter_cubic_hermite(p, dx, dy), 0.)

    return out

@ti.func
def sample_hermite_repeat_x_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+2) % width
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

    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(filter_cubic_hermite(p, dx, dy), 0.)

    return out

@ti.func
def sample_hermite_repeat_x_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+2) % width
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
    out = tm.max(filter_cubic_hermite(p, dx, dy), 0.)

    return out

@ti.func
def sample_hermite_repeat_y_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = (pos.y - 0.5) % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+2) % height

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
    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(filter_cubic_hermite(p, dx, dy), 0.)

    return out

@ti.func
def sample_hermite_repeat_y_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = (pos.y - 0.5) % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+2) % height

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
    out = tm.max(filter_cubic_hermite(p, dx, dy), 0.)

    return out

# previously - ti.real_func
@ti.func
def sample_hermite_r_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_hermite_repeat_float(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_hermite_r_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_hermite_clamp_float(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_hermite_r_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_hermite_repeat_x_float(tex, uv, repeat_w, repeat_h, window)

# previously - ti.real_func
@ti.func
def sample_hermite_r_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> float:
    return sample_hermite_repeat_y_float(tex, uv, repeat_w, repeat_h, window)



# previously - ti.real_func
@ti.func
def sample_hermite_rg_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_hermite_repeat_vec(tex, uv, repeat_w, repeat_h, window, 2))

# previously - ti.real_func
@ti.func
def sample_hermite_rg_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_hermite_clamp_vec(tex, uv, repeat_w, repeat_h, window, 2))

# previously - ti.real_func
@ti.func
def sample_hermite_rg_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_hermite_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 2))

# previously - ti.real_func
@ti.func
def sample_hermite_rg_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec2:
    return tm.vec2(sample_hermite_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 2))



# previously - ti.real_func
@ti.func
def sample_hermite_rgb_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_hermite_repeat_vec(tex, uv, repeat_w, repeat_h, window, 3))

# previously - ti.real_func
@ti.func
def sample_hermite_rgb_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_hermite_clamp_vec(tex, uv, repeat_w, repeat_h, window, 3))

# previously - ti.real_func
@ti.func
def sample_hermite_rgb_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_hermite_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 3))

# previously - ti.real_func
@ti.func
def sample_hermite_rgb_repeat_y(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec3:
    return tm.vec3(sample_hermite_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 3))


# previously - ti.real_func
@ti.func
def sample_hermite_rgba_repeat(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_hermite_repeat_vec(tex, uv, repeat_w, repeat_h, window, 4))

# previously - ti.real_func
@ti.func
def sample_hermite_rgba_clamp(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_hermite_clamp_vec(tex, uv, repeat_w, repeat_h, window, 4))

# previously - ti.real_func
@ti.func
def sample_hermite_rgba_repeat_x(tex:ti.template(),uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_hermite_repeat_x_vec(tex, uv, repeat_w, repeat_h, window, 4))

# previously - ti.real_func
@ti.func
def sample_hermite_rgba_repeat_y(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4) -> tm.vec4:
    return tm.vec4(sample_hermite_repeat_y_vec(tex, uv, repeat_w, repeat_h, window, 4))


@ti.func
def sample_b_spline_repeat_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = (pos.y - 0.5) % height

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
    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(filter_cubic_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_repeat_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = (pos.y - 0.5) % height

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
    out = tm.max(filter_cubic_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_clamp_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

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

    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(filter_cubic_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_clamp_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

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
    out = tm.max(filter_cubic_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_repeat_x_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+2) % width
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

    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])
        out[ch] = tm.max(filter_cubic_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_repeat_x_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+2) % width
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
    out = tm.max(filter_cubic_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_repeat_y_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = (pos.y - 0.5) % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+2) % height

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
    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(filter_cubic_b_spline(p, dx, dy), 0.)

    return out

@ti.func
def sample_b_spline_repeat_y_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = (pos.y - 0.5) % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+2) % height

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
    out = tm.max(filter_cubic_b_spline(p, dx, dy), 0.)

    return out


@ti.func
def sample_indexed_b_spline(
    tex:ti.template(), 
    uv:tm.vec2, 
    idx:int,
    wrap_mode:int, 
    repeat_w:int, 
    repeat_h:int, 
    window:tm.ivec4):
    width, height = int(window.z - window.x), int(window.w - window.y)

    ix, iy = tm.ivec4(0), tm.ivec4(0)
    dxdy = tm.vec2(0.)
    if wrap_mode == WrapMode.CLAMP:
        ix, iy, dxdy = dxdy_cubic_clamp(uv, width, height, repeat_w, repeat_h)
    elif wrap_mode == WrapMode.REPEAT:
        ix, iy, dxdy = dxdy_cubic_repeat(uv, width, height, repeat_w, repeat_h)
    elif wrap_mode == WrapMode.REPEAT_X:
        ix, iy, dxdy = dxdy_cubic_repeat_x(uv, width, height, repeat_w, repeat_h)
    elif wrap_mode == WrapMode.REPEAT_Y:
        ix, iy, dxdy = dxdy_cubic_repeat_y(uv, width, height, repeat_w, repeat_h)

    xofs, yofs = int(window.x), int(window.y)

    q00 = tex[idx, yofs + iy[0], xofs + ix[0]]
    q01 = tex[idx, yofs + iy[1], xofs + ix[0]]
    q02 = tex[idx, yofs + iy[2], xofs + ix[0]]
    q03 = tex[idx, yofs + iy[3], xofs + ix[0]]

    q10 = tex[idx, yofs + iy[0], xofs + ix[1]]
    q11 = tex[idx, yofs + iy[1], xofs + ix[1]]
    q12 = tex[idx, yofs + iy[2], xofs + ix[1]]
    q13 = tex[idx, yofs + iy[3], xofs + ix[1]]

    q20 = tex[idx, yofs + iy[0], xofs + ix[2]]
    q21 = tex[idx, yofs + iy[1], xofs + ix[2]]
    q22 = tex[idx, yofs + iy[2], xofs + ix[2]]
    q23 = tex[idx, yofs + iy[3], xofs + ix[2]]

    q30 = tex[idx, yofs + iy[0], xofs + ix[3]]
    q31 = tex[idx, yofs + iy[1], xofs + ix[3]]
    q32 = tex[idx, yofs + iy[2], xofs + ix[3]]
    q33 = tex[idx, yofs + iy[3], xofs + ix[3]]

    out = q11
    if ti.static(tex.n == 1):
        p = tm.mat4([
            [q00, q01, q02, q03],
            [q10, q11, q12, q13],
            [q20, q21, q22, q23],
            [q30, q31, q32, q33]
            ])
        out = tm.max(filter_cubic_b_spline(p, dx, dy), 0.)
        return out
    else:
        for ch in range(tex.n):
            p = tm.mat4([
                [q00[ch], q01[ch], q02[ch], q03[ch]],
                [q10[ch], q11[ch], q12[ch], q13[ch]],
                [q20[ch], q21[ch], q22[ch], q23[ch]],
                [q30[ch], q31[ch], q32[ch], q33[ch]]
                ])        
            out[ch] = tm.max(filter_cubic_b_spline(p, dxdy.x, dxdy.y), 0.)
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
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = (pos.y - 0.5) % height

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
    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(filter_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

@ti.func
def sample_mitchell_netravali_repeat_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = (pos.y - 0.5) % height

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
    out = tm.max(filter_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

@ti.func
def sample_mitchell_netravali_clamp_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

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

    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(filter_mitchell_netravali(p, dx, dy, b, c), 0.)
    return out

@ti.func
def sample_mitchell_netravali_clamp_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

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
    out = tm.max(filter_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

@ti.func
def sample_mitchell_netravali_repeat_x_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+2) % width
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

    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(filter_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

@ti.func
def sample_mitchell_netravali_repeat_x_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)

    x1, y1 = int(pos.x), int(pos.y)
    x0 = (x1-1) % width
    y0 = tm.min(y1-1, int(height - 1))
    x2 = (x1+1) % width
    y2 = tm.min(y1+1, int(height - 1))
    x3 = (x1+2) % width
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
    out = tm.max(filter_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

@ti.func
def sample_mitchell_netravali_repeat_y_vec(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, n:int, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = (pos.y - 0.5) % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+2) % height

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
    for ch in range(tex.n):
        p = tm.mat4([
            [q00[ch], q01[ch], q02[ch], q03[ch]],
            [q10[ch], q11[ch], q12[ch], q13[ch]],
            [q20[ch], q21[ch], q22[ch], q23[ch]],
            [q30[ch], q31[ch], q32[ch], q33[ch]]
            ])        
        out[ch] = tm.max(filter_mitchell_netravali(p, dx, dy, b, c), 0.)

    return out

@ti.func
def sample_mitchell_netravali_repeat_y_float(tex:ti.template(), uv:tm.vec2, repeat_w:int, repeat_h:int, window:tm.ivec4, b:float, c:float):
    width, height = int(window.z - window.x), int(window.w - window.y)
    hp = tm.vec2(0.5 / width, 0.5 / height)
    eps = 1e-7
    uvb = tm.vec2(0.)
    pos = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = (pos.y - 0.5) % height

    x1, y1 = int(pos.x), int(pos.y)
    x0 = tm.min(x1-1, int(width - 1))
    y0 = (y1-1) % height
    x2 = tm.min(x1+1, int(width - 1))
    y2 = (y1+1) % height
    x3 = tm.min(x1+2, int(width - 1))
    y3 = (y1+2) % height

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
    out = tm.max(filter_mitchell_netravali(p, dx, dy, b, c), 0.)

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
    uvb = (uv * tm.vec2(repeat_w, repeat_h)) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = (pos.y - 0.5) % height
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = (x0+1) % width
    y1 = (y0+1) % height
    xofs, yofs = int(window.x), int(window.y)
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
    eps = 1e-7
    uvb = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hp.x, 1. - hp.x)
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hp.y, 1. - hp.y)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.) 
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = tm.min(x0+1, int(width - 1))
    y1 = tm.min(y0+1, int(height - 1))
    xofs, yofs = int(window.x), int(window.y)
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
    eps = 1e-7
    uvb = tm.vec2(0.)
    uvb.x = (uv.x * repeat_w) % 1.
    uvb.y = tm.clamp((tm.clamp(uv.y, 0., 1. - eps) * repeat_h) % 1., hpy, 1. - hpy)
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = (pos.x - 0.5) % width
    pos.y = tm.clamp(pos.y - 0.5, 0., height - 1.)
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = (x0+1) % width
    y1 = tm.min(y0+1, int(height - 1))
    xofs, yofs = int(window.x), int(window.y)
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
    eps = 1e-7
    uvb = tm.vec2(0.)
    uvb.x = tm.clamp((tm.clamp(uv.x, 0., 1. - eps) * repeat_w) % 1., hpx, 1. - hpx)
    uvb.y = (uv.y * repeat_h) % 1.
    pos = tm.vec2(uvb.x * width, uvb.y * height)
    pos.x = tm.clamp(pos.x - 0.5, 0., width - 1.)
    pos.y = (pos.y - 0.5) % height
    x0, y0 = int(pos.x), int(pos.y)
    dx = pos.x - float(x0)
    dy = pos.y - float(y0)
    x1 = tm.min(x0+1, int(width - 1))
    y1 = (y0+1) % height
    xofs, yofs = int(window.x), int(window.y)
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
        xy, dxdy = dxdy_bilinear_clamp(uv, width, height, repeat_w, repeat_h)
    elif wrap_mode == WrapMode.REPEAT:
        xy, dxdy = dxdy_bilinear_repeat(uv, width, height, repeat_w, repeat_h)
    elif wrap_mode == WrapMode.REPEAT_X:
        xy, dxdy = dxdy_bilinear_repeat_x(uv, width, height, repeat_w, repeat_h)
    elif wrap_mode == WrapMode.REPEAT_Y:
        xy, dxdy = dxdy_bilinear_repeat_y(uv, width, height, repeat_w, repeat_h)

    xofs, yofs = int(window.x), int(window.y)
    q00 = tex[idx, yofs + xy.y, xofs + xy.x] 
    q01 = tex[idx, yofs + xy.w, xofs + xy.x] 
    q10 = tex[idx, yofs + xy.y, xofs + xy.z] 
    q11 = tex[idx, yofs + xy.w, xofs + xy.z] 
    
    q0 = tm.mix(q00, q10, dxdy.x)
    q1 = tm.mix(q01, q11, dxdy.x)

    if ti.static(tex.n == 1):
        # out = 0.
        out = tm.mix(q0, q1, dxdy.y)
        return out
    elif ti.static(tex.n == 2):
        # out = tm.vec2(0.)
        out = tm.mix(q0, q1, dxdy.y)
        return out
    elif ti.static(tex.n == 3):
        # out = tm.vec3(0.)
        out = tm.mix(q0, q1, dxdy.y)
        return out
    elif ti.static(tex.n == 4):
        # out = tm.vec4(0.)
        out = tm.mix(q0, q1, dxdy.y)
        return out