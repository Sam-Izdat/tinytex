ti module
=============
.. toctree::
    :maxdepth: 4

.. automodule:: tinytex.ti


    .. rubric:: Modes    
    .. autosummary::

        FilterMode
        WrapMode

    .. rubric:: Textures
    .. autosummary::

        Texture1D
        Texture2D
        Texture3D

    .. rubric:: Samplers
    .. autosummary::

        Sampler1D
        Sampler2D
        Sampler3D

    .. rubric:: Splines
    .. autosummary::

        compute_cubic_hermite_spline
        compute_cubic_b_spline
        compute_bc_spline

    .. rubric:: Filters
    .. autosummary::

        filter_2d_cubic_hermite
        filter_2d_cubic_b_spline
        filter_2d_mitchell_netravali

    .. rubric:: Deltas
    .. autosummary::

        dxdy_linear_clamp
        dxdy_linear_repeat
        dxdy_linear_repeat_x
        dxdy_linear_repeat_y

        dxdy_cubic_clamp
        dxdy_cubic_repeat
        dxdy_cubic_repeat_x
        dxdy_cubic_repeat_y

        dxdy_scoped_grid_linear
        dxdy_scoped_grid_cubic

    .. rubric:: Direct field sampling
    .. autosummary::

        sample_2d_indexed_bilinear
        sample_2d_indexed_b_spline

        sample_3d_nn_clamp
        sample_3d_nn_repeat
        sample_3d_nn_repeat_x
        sample_3d_nn_repeat_y
        sample_3d_nn_repeat_z

        sample_3d_trilinear_clamp
        sample_3d_trilinear_repeat
        sample_3d_trilinear_repeat_x
        sample_3d_trilinear_repeat_y
        sample_3d_trilinear_repeat_z

    .. autoclass:: FilterMode
        :members:
        :show-inheritance:

    .. autoclass:: WrapMode
        :members:
        :show-inheritance:

    .. autoclass:: Texture1D
        :members:
        :show-inheritance:

    .. autoclass:: Texture2D
        :members:
        :show-inheritance:

    .. autoclass:: Texture3D
        :members:
        :show-inheritance:

    .. autoclass:: Sampler1D
        :members:
        :show-inheritance:

    .. autoclass:: Sampler2D
        :members:
        :show-inheritance:

    .. autoclass:: Sampler3D
        :members:
        :show-inheritance:

    .. autofunction:: compute_cubic_hermite_spline
    .. autofunction:: compute_cubic_b_spline
    .. autofunction:: compute_bc_spline

    .. autofunction:: filter_2d_cubic_hermite
    .. autofunction:: filter_2d_cubic_b_spline
    .. autofunction:: filter_2d_mitchell_netravali

    .. autofunction:: dxdy_linear_clamp
    .. autofunction:: dxdy_linear_repeat
    .. autofunction:: dxdy_linear_repeat_x
    .. autofunction:: dxdy_linear_repeat_y

    .. autofunction:: dxdy_cubic_clamp
    .. autofunction:: dxdy_cubic_repeat
    .. autofunction:: dxdy_cubic_repeat_x
    .. autofunction:: dxdy_cubic_repeat_y

    .. autofunction:: dxdy_scoped_grid_linear
    .. autofunction:: dxdy_scoped_grid_cubic

    .. autofunction:: sample_2d_indexed_bilinear
    .. autofunction:: sample_2d_indexed_b_spline

    .. autofunction:: sample_3d_nn_clamp
    .. autofunction:: sample_3d_nn_repeat
    .. autofunction:: sample_3d_nn_repeat_x
    .. autofunction:: sample_3d_nn_repeat_y
    .. autofunction:: sample_3d_nn_repeat_z

    .. autofunction:: sample_3d_trilinear_clamp
    .. autofunction:: sample_3d_trilinear_repeat
    .. autofunction:: sample_3d_trilinear_repeat_x
    .. autofunction:: sample_3d_trilinear_repeat_y
    .. autofunction:: sample_3d_trilinear_repeat_z