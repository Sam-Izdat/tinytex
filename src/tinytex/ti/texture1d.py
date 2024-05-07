import typing
from typing import Union

import taichi as ti
import taichi.math as tm

import torch
import numpy as np
from enum import IntEnum    

from .prep import *

@ti.data_oriented
class Texture1D:
    """
    1D read-write texture. Can be initialized with either texture shape or texture data.
    """
    def __init__():
        pass