tinytex
=======

Python texture sampling, processing and synthesis library for PyTorch-involved projects.

This library is a hodgepodge of tangentially-related procedures useful for sampling, creating and 
modifying various kinds of textures. This is primarily intended for batched or unbatched 
PyTorch image tensors. This library provides:

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
- pseudo-random number generation
- generating tiling spatial-domain noise
- generating spectral-domain noise
- warping image coordinates
- transforming 1D and 2D images to and from Haar wavelet coefficients
- (experimental) backend-agnostic 1D/2D/3D textures for Taichi (if installed with Taichi optional dependency)
    
  - load from and save to the filesystem
  - convert textures to and from PyTorch tensors
  - sample textures with lower or higher-order interpolation/approximation

Getting started
---------------

* Run :code:`pip install tinytex`
* Run :code:`ttex-setup` 

About 
-----

.. only:: html

    .. hlist::
        :columns: 4
    
        * v |release|
        * `PDF manual <./tinytex.pdf>`_
        * `Previous versions <https://github.com/Sam-Izdat/tinytex/releases>`_
        * :doc:`Release notes <source/about_release_notes>`

.. rubric:: License

:doc:`MIT License <source/license>` on all original code - see source for details

Reference
---------

.. only:: html

    See: :doc:`reference section <source/tinytex>`.

.. toctree::
    :maxdepth: 2
    :caption: Reference:
    :hidden:

    source/tinytex
    source/about_release_notes
    source/license
    genindex


Links
-----

* `GitHub <https://github.com/Sam-Izdat/tinytex>`_
* `PyPi <https://pypi.org/project/tinytex/>`_

.. toctree::
    :maxdepth: 2
    :caption: Links:
    :hidden:

    GitHub <https://github.com/Sam-Izdat/tinytex>
    PyPi <https://pypi.org/project/tinytex/>
    Docs <https://sam-izdat.github.io/tinytex/>

Sibling projects
----------------

* `tinycio <https://sam-izdat.github.io/tinycio/>`_
* `tinyfilm <https://sam-izdat.github.io/tinyfilm/>`_

.. toctree::
    :maxdepth: 2
    :caption: Sibling projects:
    :hidden:

    tinycio <https://sam-izdat.github.io/tinycio/>
    tinyfilm <https://sam-izdat.github.io/tinyfilm/>