Work with surface geometry
==========================

The :class:`.SurfaceOps` class provides tools for converting and processing normal maps, height maps, curvature, and ambient occlusion in a y-up OpenGL-style tangent space.

Convert normal maps to angle maps:

.. code-block:: python

    from tinytex import SurfaceOps, fsio

    normal_map = fsio.load_image("normal.png")
    angle_map = SurfaceOps.normals_to_angles(normal_map, normalize=True, rescaled=True)
    fsio.save_image(angle_map, "angles.png")

Convert angles back to normals:

.. code-block:: python

    normals = SurfaceOps.angles_to_normals(angle_map, normalize=True, rescaled=True)
    fsio.save_image(normals, "reconstructed_normals.png")

Reorient/blend detail normals onto base:

.. code-block:: python

    detail = fsio.load_image("detail_normals.png")
    blended = SurfaceOps.blend_normals(normal_map, detail, rescaled=True)
    fsio.save_image(blended, "blended.png")

Generate normals from height:

.. code-block:: python

    height = fsio.load_image("height.png")
    normals = SurfaceOps.height_to_normals(height, rescaled=True)
    fsio.save_image(normals, "generated_normals.png")

Generate height from normals:

.. code-block:: python

    height, scale = SurfaceOps.normals_to_height(normal_map, self_tiling=True, rescaled=True)
    fsio.save_image(height, "reconstructed_height.png")

Estimate curvature from height:

.. code-block:: python

    curvature, cavities, peaks = SurfaceOps.height_to_curvature(height)
    fsio.save_image(curvature, "curvature.png")

Compute screen space AO and bent normals:

.. code-block:: python

    ao, bent = SurfaceOps.compute_occlusion(height_map=height, normal_map=normals, rescaled=True)
    fsio.save_image(ao, "ao.png")
    fsio.save_image(bent, "bent_normals.png")

Recompute z channel for tangent-space normals:

.. code-block:: python

    fixed = SurfaceOps.recompute_z(normal_map, rescaled=True)
    fsio.save_image(fixed, "fixed_normals.png")

See: :class:`.SurfaceOps`