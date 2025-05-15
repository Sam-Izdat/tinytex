Use smoothstep
==============

Stateful usage
--------------

Create a reusable interpolator instance:

.. code-block:: python

    from tinytex import Smoothstep

    # cubic smoothstep
    s = Smoothstep('cubic_polynomial')
    y = s.forward(0.0, 1.0, x)
    x_back = s.inverse(y)

    # rational smoothstep of order 4
    r = Smoothstep('rational', n=4)
    y2 = r.forward(0.0, 1.0, x)

Stateless (one-off) usage
-------------------------

Call directly on the class:

.. code-block:: python

    from tinytex import Smoothstep

    y = Smoothstep.apply('quintic_polynomial', 0.0, 1.0, x)
    y3 = Smoothstep.apply('rational', 0.0, 1.0, x, n=5)

Normalize inputs manually
-------------------------

If you need to clamp or remap before interpolation:

.. code-block:: python

    from tinytex import Smoothstep

    x_norm = Smoothstep._normalize(x, edge0, edge1)

See: :class:`.Smoothstep`