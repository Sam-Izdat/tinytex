Create texture atlases
======================

The :class:`.Atlas` class packs multiple image tensors into a single texture atlas. 

.. code-block:: python

    from tinycio import fsio
    from tinytex import Atlas

Load and pack all images from a directory:

.. code-block:: python

    atlas = Atlas.from_dir(
        path='assets/',
        ext='.png',
        channels=3,
        allow_mismatch=True,
        max_h=1024,
        max_w=1024,
        crop=True,
        row=False,
        sort='height'
    )

Save the packed atlas:

.. code-block:: python

    fsio.save_image(atlas.atlas, 'atlas.png')

Sample a texture by name or index:

.. code-block:: python

    tex = atlas.sample('stone')
    tex2 = atlas.sample(0)

    fsio.save_image(tex, 'stone_out.png')

Sample randomly:

.. code-block:: python

    rand_tex = atlas.sample_random()
    fsio.save_image(rand_tex, 'rand.png')

Generate a tiling mask using randomly overlaid textures:

.. code-block:: python

    mask = atlas.generate_mask(
        shape=(512, 512),
        scale=0.5,
        samples=3
    )

    fsio.save_image(mask, 'mask.png')

Manual construction is also possible:

.. code-block:: python

    im1 = fsio.load_image('a.png')
    im2 = fsio.load_image('b.png')

    atlas = Atlas(min_size=256, max_size=2048, force_square=True)
    atlas.add('a', im1)
    atlas.add('b', im2)
    atlas.pack(crop=True)

    fsio.save_image(atlas.atlas, 'manual.png')

To inspect bounds:

.. code-block:: python

    print(atlas.index['a'])  # -> (x0, y0, x1, y1)

The packing is either rectangular (default) or row-based (`row=True`), depending on your layout needs. Use rectangular unless everything’s the same height.

See: :class:`.Atlas`