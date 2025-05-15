Make noise
==========

Spatial-domain noise
--------------------

Make Perlin noise:

.. code-block:: python

    from tinytex import SpatialNoise

    img = SpatialNoise.perlin((256, 256), density=8.)

Make noise from fractal layers:

.. code-block:: python

    img = SpatialNoise.fractal((256, 256), density=4., octaves=6)

Make turbulent noise:

.. code-block:: python

    img = SpatialNoise.turbulence((256, 256), density=6., ridge=True)

Make Worley noise:

.. code-block:: python

    img = SpatialNoise.worley((256, 256), density=10.)

Spectral-domain noise
---------------------

Generate flat-spectrum white noise:

.. code-block:: python

    from tinytex import SpectralNoise

    img = SpectralNoise.white(256, 256)

Generate other types of colored noise

.. code-block:: python

    img = SpectralNoise.pink(256, 256)
    img = SpectralNoise.brownian(256, 256)
    img = SpectralNoise.blue(256, 256)
    img = SpectralNoise.violet(256, 256)


Define your own power spectrum for custom spectral shaping:

.. code-block:: python

    import torch
    from tinytex import SpectralNoise

    def bandpass(f):
        return torch.exp(-((f - 0.2)**2) / 0.01)

    img = SpectralNoise.noise_psd_2d(256, 256, psd=bandpass)

See: :class:`.SpatialNoise`, :class:`.SpectralNoise`