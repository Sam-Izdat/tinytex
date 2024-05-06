from enum import IntEnum

class FilterMode(IntEnum):
    """Texture filter mode"""
    NEAREST             = 1<<0
    BILINEAR            = 1<<1
    TRILINEAR           = 1<<2
    HERMITE             = 1<<3 
    B_SPLINE            = 1<<4 
    MITCHELL_NETRAVALI  = 1<<5 
    CATMULL_ROM         = 1<<6 

    SUPPORTED_2D    = NEAREST | BILINEAR | HERMITE | B_SPLINE | MITCHELL_NETRAVALI | CATMULL_ROM
    SUPPORTED_3D    = NEAREST | TRILINEAR
    SUPPORTED_GRID  = NEAREST | BILINEAR | B_SPLINE

class WrapMode(IntEnum):
    """Texture wrap mode"""
    REPEAT      = 1<<0
    CLAMP       = 1<<1
    REPEAT_X    = 1<<2
    REPEAT_Y    = 1<<3
    REPEAT_Z    = 1<<4

    # TODO: MIRROR, etc

    SUPPORTED_2D    = REPEAT | CLAMP | REPEAT_X | REPEAT_Y
    SUPPORTED_3D    = REPEAT | CLAMP # | REPEAT_X | REPEAT_Y | REPEAT_Z # TODO: Extended 3D repeat modes
    SUPPORTED_GRID  = REPEAT | CLAMP
