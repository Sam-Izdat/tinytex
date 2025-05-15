Work with tiles
===============

The :class:`.Tiling` class handles image tiling, seamless merging, and Poisson-based tile blending. This is useful for working with large textures or tiled procedural data.

Splitting and merging
---------------------

Split a large image into smaller tiles:

.. code-block:: python
    
    from tinytex import Tiling

    tiles, rows, cols = Tiling.split(image, shape=(64, 64))

You can merge the tiles back:

.. code-block:: python

    merged = Tiling.merge(tiles, rows, cols)

Tiles are ordered row-major (left-to-right, top-to-bottom):

.. code-block:: text

    | 0 | 1 | 2 |
    -------------
    | 3 | 4 | 5 |
    -------------
    | 6 | 7 | 8 |

Tile indexing helpers
---------------------

Use these if you're doing custom traversal or adjacency logic:

.. code-block:: python

    row, col = Tiling.get_tile_position(idx, cols)
    idx = Tiling.get_tile_index(row, col, cols)
    neighbors = Tiling.get_tile_neighbors(row, col, rows, cols, wrap=True)

Blending tiles
--------------

Blend adjacent tiles to remove seams between them:

.. code-block:: python

    blended = Tiling.blend(tiles, rows=rows, cols=cols)

This uses a Poisson solver to match gradients across tile edges. It’s expensive but high quality.

For self-tiling output (no visible borders when tiled):

.. code-block:: python

    seamless = Tiling.blend(tiles, rows, cols, wrap=True)

If your tiles represent **vector data** (e.g., normal maps), pass `vector_data=True` to convert vectors into angles during blending:

.. code-block:: python

    seamless = Tiling.blend(normal_tiles, rows, cols, vector_data=True)

.. note::
    
    This is a heavy operation. By default it uses SciPy's solver, but will auto-switch to PyAMG or AMGCL if available.

See: :class:`.Tiling`