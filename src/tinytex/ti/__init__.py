"""
Taichi texture sampling module. Supports CPU, CUDA and Vulkan backends.

.. note::

	This module is meant to be a relatively backend-agnostic API with extended capabilities and so 
	does not (explicitly) use hardware-native texture sampling. This may incur a performance cost.
"""
from .params import FilterMode, WrapMode
from .splines import filter_cubic_hermite, filter_cubic_b_spline, filter_mitchell_netravali, \
	compute_cubic_hermite_spline, compute_cubic_b_spline, compute_bc_spline
from .texture2d import Texture2D
# from .texture3d import Texture3D
from .sampler2d import Sampler2D, \
	sample_indexed_bilinear, sample_indexed_b_spline, \
	dxdy_2D_scoped_grid_cubic, dxdy_2D_scoped_grid_bilinear
from .sampler3d import Sampler3D