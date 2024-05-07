from enum import IntEnum

class FilterMode(IntEnum):
    """
    Texture filter mode.

    .. list-table:: Available Filter Modes:
        :widths: 15 70 5 5 5
        :header-rows: 1

        * - Identifier
          - Description
          - 1D
          - 2D
          - 3D
        * - NEAREST
          - Nearest neighbor - point sampling.
          - ✓
          - ✓
          - ✓

        * - LINEAR
          - Linear interpolation.
          - ✓
          - 
          - 
        * - BILINEAR
          - Bilinear interpolation.
          -
          - ✓
          - 
        * - TRILINEAR
          - Trilinear interpolation.
          -
          - 
          - ✓
        * - HERMITE
          - Cubic Hermite (bicubic) interpolation.
          - 
          - ✓
          - 
        * - B_SPLINE
          - Cubic B-spline approximation.
          - 
          - ✓
          - 
        * - MITCHELL_NETRAVALI
          - Mitchell-Netravali interp/approx.
          - 
          - ✓
          - 
        * - CATMULL_ROM
          - Catmull-Rom interpolation.
          -
          - ✓
          - 
    """
    NEAREST             = 1<<0
    LINEAR              = 1<<1
    BILINEAR            = 1<<2
    TRILINEAR           = 1<<3
    HERMITE             = 1<<4 
    B_SPLINE            = 1<<5 
    MITCHELL_NETRAVALI  = 1<<6 
    CATMULL_ROM         = 1<<7 

    SUPPORTED_1D    = NEAREST | LINEAR
    SUPPORTED_2D    = NEAREST | BILINEAR | HERMITE | B_SPLINE | MITCHELL_NETRAVALI | CATMULL_ROM
    SUPPORTED_3D    = NEAREST | TRILINEAR

class WrapMode(IntEnum):
    """
    Texture wrap mode.

    .. list-table:: Available Wrap Modes:
        :widths: 15 70 5 5 5
        :header-rows: 1

        * - Identifier
          - Description
          - 1D
          - 2D
          - 3D
        * - REPEAT
          - Repeat all dimensions.
          - ✓
          - ✓
          - ✓
        * - CLAMP
          - Clamp all dimentions.
          - ✓
          - ✓
          - ✓
        * - REPEAT_X
          - Repeat x/width only.
          - ✓
          - ✓
          - ✓
        * - REPEAT_Y
          - Repeat y/height only.
          - 
          - ✓
          - ✓
        * - REPEAT_Z
          - Repeat z/depth only.
          - 
          -
          - ✓
    """
    REPEAT      = 1<<0
    CLAMP       = 1<<1
    REPEAT_X    = 1<<2
    REPEAT_Y    = 1<<3
    REPEAT_Z    = 1<<4

    # TODO: MIRROR, etc

    SUPPORTED_1D    = REPEAT | CLAMP | REPEAT_X
    SUPPORTED_2D    = REPEAT | CLAMP | REPEAT_X | REPEAT_Y
    SUPPORTED_3D    = REPEAT | CLAMP | REPEAT_X | REPEAT_Y | REPEAT_Z
