import typing
from typing import Union

import taichi as ti
import taichi.math as tm

@ti.func
def compute_cubic_hermite_spline(p, x:float) -> float:
    """
    Compute cubic Hermite spline. Interpolating spline.

    :param p: Four control points of interpolation.
    :type p: taichi.math.vec4
    :param x: Spline curve parameter.
    """
    # # Alternative formulation:
    # x_squared = x**2
    # x_cubed = x**3
    # out = (-p[0]/2.0 + (3.0*p[1])/2.0 - (3.0*p[2])/2.0 + p[3]/2.0) * x_cubed \
    #     + (p[0] - (5.0*p[1])/2.0 + 2.0*p[2] - p[3] / 2.0) * x_squared \
    #     + (-p[0]/2.0 + p[2]/2.0) * x + p[1]
    # return out
    return p[1] + 0.5 * x*(p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0])))

@ti.func
def compute_cubic_b_spline(p:tm.vec4, x:float) -> float:
    """
    Compute B-spline. Non-interpolating spline.

    :param p: Four control points of approximation.
    :type p: taichi.math.vec4
    :param x: Spline curve parameter.
    """
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
def compute_bc_spline(p:tm.vec4, x:float, b:float, c:float) -> float:
    """
    Compute BC-spline. Non-interpolating spline unless B=0.

    :param p: Four control points of approximation.
    :type p: taichi.math.vec4
    :param x: Spline curve parameter.
    :param b: BC-spline B-value.
    :param c: BC-spline C-value.
    """
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
def filter_cubic_hermite(p:tm.mat4, x:float, y:float) -> float:
    """
    Cubic Hermite filter. Interpolates 4x4 point samples.
    
    :param p: 4x4 samples of approximation.
    :type p: taichi.math.vec4
    :param x: x-delta
    :param y: y-delta
    """
    arr = tm.vec4(0.)
    arr[0] = compute_cubic_hermite_spline(tm.vec4(p[0,:]), y)
    arr[1] = compute_cubic_hermite_spline(tm.vec4(p[1,:]), y)
    arr[2] = compute_cubic_hermite_spline(tm.vec4(p[2,:]), y)
    arr[3] = compute_cubic_hermite_spline(tm.vec4(p[3,:]), y)
    return compute_cubic_hermite_spline(arr, x)

@ti.func
def filter_cubic_b_spline(p:tm.mat4, x:float, y:float) -> float:
    """
    Cubic B-spline filter. Approximates 4x4 point samples.

    :param p: 4x4 samples of approximation.
    :type p: taichi.math.vec4
    :param x: x-delta
    :param y: y-delta
    """
    arr = tm.vec4(0.)
    arr[0] = compute_cubic_b_spline(tm.vec4(p[0,:]), y)
    arr[1] = compute_cubic_b_spline(tm.vec4(p[1,:]), y)
    arr[2] = compute_cubic_b_spline(tm.vec4(p[2,:]), y)
    arr[3] = compute_cubic_b_spline(tm.vec4(p[3,:]), y)
    return compute_cubic_b_spline(arr, x)

@ti.func
def filter_mitchell_netravali(p:tm.mat4, x:float, y:float, b:float, c:float) -> float:
    """
    Mitchell-Netravali/BC-spline filter. Approximates/interpolates 4x4 point samples.

    :param p: 4x4 samples of approximation.
    :type p: taichi.math.vec4
    :param x: x-delta
    :param y: y-delta
    :param b: BC-spline B-value.
    :param c: BC-spline C-value.
    """
    arr = tm.vec4(0.)
    arr[0] = compute_bc_spline(tm.vec4(p[0,:]), y, b, c)
    arr[1] = compute_bc_spline(tm.vec4(p[1,:]), y, b, c)
    arr[2] = compute_bc_spline(tm.vec4(p[2,:]), y, b, c)
    arr[3] = compute_bc_spline(tm.vec4(p[3,:]), y, b, c)
    return compute_bc_spline(arr, x, b, c)