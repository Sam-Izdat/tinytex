![tinytex](https://raw.githubusercontent.com/Sam-Izdat/tinytex/main/doc/images/tinytex_sm.png)

* [Project site & docs](https://sam-izdat.github.io/tinytex/) 
* [PyPi](https://pypi.org/project/tinytex/)

# About

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


# Getting started

* Recommended: set up a clean Python environment
* [Install PyTorch  as instructed here](https://pytorch.org/get-started/locally/)
* Run  `pip install tinytex`
* Run  `ttex-setup` ([iio docs on fi](https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.freeimage.html#module-imageio.plugins.freeimage))

[See the docs](https://sam-izdat.github.io/tinytex-docs/) for the rest.