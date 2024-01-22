tinytex
=======

A lightweight Python texture editing and generation library: surface processing, atlasing, tiling, signed distance fields, resampling and noise generation.


TODO: polar, log-polar transform, wavelets, chromatic PSF & conv, SH

This library can be used to:

- resample images
- create texture atlases
- split textures into tiles and merge tile sets
- seamlessly stitch textures with color or vector data

    - tile set blending
    - self-tiling 

- compute/approximate surface geometry

    - normals from height/displacement
    - height/displacement from normals
    - curvature

- approximate occlusion

    - screen space ambient occlusion 
    - bent normals

- blend normal maps
- convert between specular-workflow and metallic-workflow PBR texture maps
- generate tiling noise

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