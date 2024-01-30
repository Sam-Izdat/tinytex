tinytex
=======

A lightweight Python texture editing and texture synthesis library.

This library can be used to:

- resample images
- create and sample texture atlases
- split and merge tiles
- seamlessly stitch textures with color or vector data for mutual or self-tiling
- compute and render 2D signed distance fields
- compute and approximate surface geometry (normals/displacement/curvature)
- approximate occlusion and bent normals
- blend normal maps
- generate tiling noise
- generate tiling masks from texture atlases
- transform images to and from log-polar coordinate space
- convert images to and from Haar wavelet coefficients
- point spread functions and approximate aperture diffraction

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

.. rubric:: Optional requirements

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

    tinylcm <https://sam-izdat.github.io/tinylcm-docs/>
    tinypbr <https://sam-izdat.github.io/tinypbr-docs/>
    tinycio <https://sam-izdat.github.io/tinycio-docs/>