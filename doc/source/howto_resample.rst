Resample images
===============

The :class:`.Resampling` class works on PyTorch tensors of shape ``[C, H, W]`` or ``[N, C, H, W]``. 

Tile an image to a target size:

.. code-block:: python

    from tinytex import Resampling, fsio
    import torch

    img = fsio.load_image("input.png")      # shape [C, H, W]
    tiled = Resampling.tile(img, (512, 512))
    fsio.save_image(tiled, "tiled.png")

Tile by a fixed repeat count:

.. code-block:: python

    repeated = Resampling.tile_n(img, repeat_h=4, repeat_w=2)
    fsio.save_image(repeated, "repeated.png")

Make a square by tiling:

.. code-block:: python

    square = Resampling.tile_to_square(img, target_size=256)
    fsio.save_image(square, "square.png")

Crop to a box:

.. code-block:: python

    cropped = Resampling.crop(img, shape=(100, 150), start=(10, 20))
    fsio.save_image(cropped, "cropped.png")

Resize to exact dimensions:

.. code-block:: python

    # simple bilinear down or up-sampling
    resized = Resampling.resize(img, (200, 300), mode="bilinear")
    fsio.save_image(resized, "resized.png")

Iterative downsample:

.. code-block:: python

    small = Resampling.resize(img, (64, 64), mode="bilinear", iterative_downsample=True)
    fsio.save_image(small, "small.png")

Area downsample:

.. code-block:: python

    small = Resampling.resize(img, (64, 64), mode="area", iterative_downsample=False)
    fsio.save_image(small, "small.png")

Aspect-preserving shortest-edge resize:

.. code-block:: python

    se = Resampling.resize_se(img, size=128, mode="bicubic")
    fsio.save_image(se, "short_edge.png")

Aspect-preserving longest-edge resize:

.. code-block:: python

    le = Resampling.resize_le(img, size=256, mode="bicubic")
    fsio.save_image(le, "long_edge.png")

Resize longest edge to next power-of-two:

.. code-block:: python

    pot = Resampling.resize_le_to_next_pot(img, mode="bicubic")
    fsio.save_image(pot, "pot.png")

Pad right & bottom:

.. code-block:: python

    padded = Resampling.pad_rb(img, shape=(300,300), mode="replicate")
    fsio.save_image(padded, "padded.png")

Pad up to square power-of-two:

.. code-block:: python

    square_pot = Resampling.pad_to_next_pot(img, mode="replicate")
    fsio.save_image(square_pot, "square_pot.png")

Generate a mip pyramid tensor:

.. code-block:: python

    # builds tensor sized [C, H, W + W//2]
    pyramid = Resampling.generate_mip_pyramid(img)
    # you can inspect each level with compute_lod_offsets:
    offsets = Resampling.compute_lod_offsets(img.size(1))

Sample a specific LOD with bilinear filtering:

.. code-block:: python

    # lod = 0 (base), 1, 2… can be fractional
    out0 = Resampling.sample_lod_bilinear(pyramid, 128, 128, lod=1.5)
    fsio.save_image(out0, "lod_bilinear.png")

Sample a specific LOD with 4-tap B-spline:

.. code-block:: python

    out1 = Resampling.sample_lod_bspline_hybrid(pyramid, 128, 128, lod=2.3)
    fsio.save_image(out1, "lod_bspline.png")

Sample a specific LOD with dithered B-spline:

.. code-block:: python

    out2 = Resampling.sample_lod_bspline_dither(pyramid, 128, 128, lod=2.3)
    fsio.save_image(out2, "lod_dither.png")

See: :class:`.Resampling` 