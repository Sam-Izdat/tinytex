tinytex
=======

Python texture sampling, processing and synthesis library for PyTorch-involved projects.

This library is a hodgepodge of tangentially-related procedures useful for sampling, making and 
modifying various kinds of textures. The primary input and output is batched or unbatched 
PyTorch image tensors. This library provides:

- backend-agnostic 1D/2D/3D textures for Taichi (if installed with Taichi optional requirement)
    
  - load from and save to the filesystem
  - convert textures to and from PyTorch tensors
  - sample textures with lower or higher-order interpolation/approximation

- image resampling/rescaling, cropping and padding
- tiling

  - split images into tiles 
  - merge tiles back into images
  - seamlessly stitch textures with color or vector data for mutual tiling or self-tiling

- texture atlases

  - pack images into texture atlases
  - sample images from texture atlases
  - generate tiling masks from texture atlases

- computing and rendering 2D signed distance fields
- computing and approximating surface geometry 

  - normals to height
  - height to normals 
  - height/normals to curvature

- approximating ambient occlusion and bent normals
- blending multiple normal maps
- generating tiling noise
- warping image coordinates
- transforming 1D and 2D images to and from Haar wavelet coefficients
- computing chromatic point spread functions and approximating aperture diffraction

Getting started
---------------

* Run :code:`pip install tinytex`
* Run :code:`ttex-setup` 

About 
-----
Release version 
|version|
|release|

.. rubric:: Requirements

- PyTorch >=2.0 (earlier versions untested)
- NumPy >=1.21
- imageio >=2.9 (with PNG-FI FreeImage plugin)
- tqdm >=4.64
- toml >=0.10
- tinycio

.. rubric:: License

:doc:`MIT License <source/license>` on all original code - see source for details

Special thanks
--------------

* Placeholder

.. toctree::
    :maxdepth: 2
    :caption: Reference:
    :hidden:

    source/tinytex
    source/about_release_notes
    source/license
    genindex

.. source/about_modules
.. source/tinytex

.. toctree::
    :maxdepth: 2
    :caption: Links:
    :hidden:

    GitHub <https://github.com/Sam-Izdat/tinytex>
    PyPi <https://pypi.org/project/tinytex/>
    Docs <https://sam-izdat.github.io/tinytex-docs/>

.. toctree::
    :maxdepth: 2
    :caption: Sibling projects:
    :hidden:

    tinycio <https://sam-izdat.github.io/tinycio-docs/>
    tinypbr <https://sam-izdat.github.io/tinypbr-docs/>
    tinylcm <https://sam-izdat.github.io/tinylcm-docs/>
    tinysimi <https://sam-izdat.github.io/tinysimi-docs/>
    tinytrace <https://sam-izdat.github.io/tinytrace-docs/>
    tinyraster <https://sam-izdat.github.io/tinyraster-docs/>