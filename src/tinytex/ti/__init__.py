"""
Taichi texture sampling module. Supports CPU, CUDA and Vulkan backends.

.. note::

	This module is meant to be a backend-agnostic API with extended sampling capabilities and so 
	does not (explicitly) use hardware-native texture sampling. This may incur a performance cost. 
	Further optimzation `is also possible <https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering>`_.
	While this benefits immensely from hardware acceleration, it is not meant to be a hyper-optimized solution.

.. note::

	DRY is, to some extent, forsaken because persuing it here can lead to kernel compilation issues.
	If extending this to higher-dimensional textures with higher-order interpolation/approximation, 
	an additional layer of abstraction would likely be a sensible idea.
"""
from .params import FilterMode, WrapMode
from .splines import filter_2d_cubic_hermite, filter_2d_cubic_b_spline, filter_2d_mitchell_netravali, \
	compute_cubic_hermite_spline, compute_cubic_b_spline, compute_bc_spline
from .texture1d import Texture1D
from .texture2d import Texture2D
from .texture3d import Texture3D

from .sampler1d import Sampler1D
from .sampler2d import Sampler2D, \
	sample_2d_bilinear_clamp, sample_2d_bilinear_repeat, sample_2d_bilinear_repeat_x, sample_2d_bilinear_repeat_y, \
	dxdy_linear_clamp, dxdy_linear_repeat, dxdy_linear_repeat_x, dxdy_linear_repeat_y, \
	dxdy_cubic_clamp, dxdy_cubic_repeat, dxdy_cubic_repeat_x, dxdy_cubic_repeat_y, \
	sample_2d_indexed_bilinear, sample_2d_indexed_b_spline, \
	dxdy_scoped_grid_cubic, dxdy_scoped_grid_linear
from .sampler3d import Sampler3D, \
	sample_3d_trilinear_clamp, sample_3d_trilinear_repeat, sample_3d_trilinear_repeat_x, sample_3d_trilinear_repeat_y, sample_3d_trilinear_repeat_z, \
	sample_3d_nn_clamp, sample_3d_nn_repeat, sample_3d_nn_repeat_x, sample_3d_nn_repeat_y, sample_3d_nn_repeat_z