Create SDFs
===========

The :class:`.SDF` class provides signed distance field computation, primitive shape generation, and rendering. This is all 'borrowed' from `"2D SDF functions" by Inigo Quilez <https://iquilezles.org/articles/distfunctions2d/>`_.

Generating basic shapes
-----------------------

Create a soft circular mask:

.. code-block:: python
    
    from tinytex import SDF

    sdf = SDF.circle(size=128, radius=40)
    img = SDF.render(sdf, shape=(128, 128), edge0=0.4, edge1=0.6)

Create a rectangular SDF:

.. code-block:: python

    sdf = SDF.box(size=128, box_shape=(64, 32))
    img = SDF.render(sdf, shape=(128, 128), edge0=0.45, edge1=0.5)

Create a segment-shaped SDF:

.. code-block:: python

    sdf = SDF.segment(size=128, a=(32, 32), b=(96, 96))
    img = SDF.render(sdf, shape=(128, 128))

Converting images to SDF
------------------------

Convert a binary mask into a normalized SDF:

.. code-block:: python

    binary_mask = torch.rand(1, 128, 128) > 0.5
    sdf = SDF.compute(binary_mask.float(), threshold=0.5)
    img = SDF.render(sdf, shape=(128, 128))

Tiling for seamless textures:

.. code-block:: python

    sdf = SDF.circle(size=64, radius=20, tile_to=128)
    img = SDF.render(sdf, shape=(128, 128))

Combining fields
----------------

You can blend shapes via min/max:

.. code-block:: python

    a = SDF.circle(64, radius=20)
    b = SDF.box(64, box_shape=(32, 32))
    union = SDF.min(a, b)
    intersect = SDF.max(a, b)
    img = SDF.render(union, shape=(64, 64))

See: :class:`.SDF`