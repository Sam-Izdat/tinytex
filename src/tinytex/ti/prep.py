import typing
from typing import Union

import taichi as ti
import taichi.math as tm

import torch
import numpy as np   

# This "prep_x" stuff is ugly and redundant, but this is easier with some special handling.
#
# Valid input data here can be any of: 
# - [C, H, W] torch tensor
# - [H, W] torch tensor if C=1
# - [C] torch tensor (single color value)
# - [H, W, C] numpy array
# - [H, W] numpy array if C=1
# - [C] numpy array (single color value)
# - float, tm.vec2, tm.vec3, or tm.vec4 value
# - other numeric value if C=1

def prep_r(val:torch.Tensor, flip_y:bool = False):
    """Converts image data to [H, W] floating point torch image tensor"""
    if torch.is_tensor(val) and val.dim() == 3:
        if val.size(0) != 1: val = val[0:1]
        val = val.permute(1, 2, 0)
    elif torch.is_tensor(val) and val.dim() == 2:
        val = val.unsqueeze(-1)
    elif torch.is_tensor(val) and val.dim() == 1 and val.size(0) == 1:
        val = val.unsqueeze(0)
    elif isinstance(val, np.ndarray) and len(val.shape) == 3 and val.shape[2] == 1:
        val = torch.from_numpy(val.squeeze(-1))
    elif isinstance(val, np.ndarray) and len(val.shape) == 2 and val.shape[2] == 1:
        val = torch.from_numpy(val)
    elif type(val) == int or type(val) == float or (type(val) == str and isnumber(val)):
        val = torch.tensor([[float(val)]], dtype=torch.float32)
    else: 
        raise Exception("Expected [C=1, H, W] image tensor, [H, W, C=1] ndarray or float value")
    if flip_y: val = torch.flip(val, dims=[0])
    return val.float()

def prep_rg(val:torch.Tensor, flip_y:bool = False):
    """Converts image data to [H, W, C=2] floating point torch image tensor"""
    if torch.is_tensor(val) and val.dim() == 3 and val.size(0) == 2:
        val = val.permute(1, 2, 0) # C, H, W -> H, W, C
    elif torch.is_tensor(val) and val.dim() == 1 and val.size(0) == 2:
        val = val.unsqueeze(0).unsqueeze(0)
    elif isinstance(val, np.ndarray) and len(val.shape) == 3 and val.shape[2] == 2:
        val = torch.from_numpy(val)
    elif isinstance(val, np.ndarray) and len(val.shape) == 1 and val.shape[0] == 2:
        val = torch.from_numpy(val).unsqueeze(0).unsqueeze(0)
    elif (type(val) == ti.lang.matrix.Vector or type(val) == ti.lang.matrix.Matrix) and len(val) == 2:
        val = torch.tensor([[[val.r, val.g, val.b]]], dtype=torch.float32)
    else: raise Exception("Expected [C=2, H, W] image tensor, [H, W, C=2] ndarray or vec2 value")
    if flip_y: val = torch.flip(val, dims=[0])
    return val.float()

def prep_rgb(val:torch.Tensor, flip_y:bool = False):
    """Converts image data to [H, W, C=3] floating point torch image tensor"""
    if torch.is_tensor(val) and val.dim() == 3 and val.size(0) == 3:
        val = val.permute(1, 2, 0) # C, H, W -> H, W, C
    elif torch.is_tensor(val) and val.dim() == 1 and val.size(0) == 3:
        val = val.unsqueeze(0).unsqueeze(0)
    elif isinstance(val, np.ndarray) and len(val.shape) == 3 and val.shape[2] == 3:
        val = torch.from_numpy(val)
    elif isinstance(val, np.ndarray) and len(val.shape) == 1 and val.shape[0] == 3:
        val = torch.from_numpy(val).unsqueeze(0).unsqueeze(0)
    elif (type(val) == ti.lang.matrix.Vector or type(val) == ti.lang.matrix.Matrix) and len(val) == 3:
        val = torch.tensor([[[val.r, val.g, val.b]]], dtype=torch.float32)
    else: raise Exception("Expected [C=3, H, W] image tensor, [H, W, C=3] ndarray or vec3 value")
    if flip_y: val = torch.flip(val, dims=[0])
    return val.float()

def prep_rgba(val:torch.Tensor, flip_y:bool = False):
    """Converts image data to [H, W, C=4] floating point torch image tensor"""
    if torch.is_tensor(val) and val.dim() == 3 and val.size(0) == 4:
        val = val.permute(1, 2, 0) # C, H, W -> H, W, C
    elif torch.is_tensor(val) and val.dim() == 1 and val.size(0) == 4:
        val = val.unsqueeze(0).unsqueeze(0)
    elif isinstance(val, np.ndarray) and len(val.shape) == 3 and val.shape[2] == 4:
        val = torch.from_numpy(val)
    elif isinstance(val, np.ndarray) and len(val.shape) == 1 and val.shape[0] == 4:
        val = torch.from_numpy(val).unsqueeze(0).unsqueeze(0)
    elif (type(val) == ti.lang.matrix.Vector or type(val) == ti.lang.matrix.Matrix) and len(val) == 4:
        val = torch.tensor([[[val.r, val.g, val.b]]], dtype=torch.float32)
    else: raise Exception("Expected [C=4, H, W] image tensor, [H, W, C=4] ndarray or vec4 value")
    if flip_y: val = torch.flip(val, dims=[0])
    return val.float()

def count_channels(im:torch.Tensor):
    channels = 0
    if   isinstance(im, float): channels = 1
    elif isinstance(im, torch.Tensor) and torch.is_floating_point(im):
        if   im.dim() == 1 and im.size(0) == 1: channels = 1
        elif im.dim() == 2 or (im.dim() == 3 and im.size(0) == 1): channels = 1
        elif (im.dim() == 3 or im.dim() == 1) and im.size(0) == 2: channels = 2
        elif (im.dim() == 3 or im.dim() == 1) and im.size(0) == 3: channels = 3
        elif (im.dim() == 3 or im.dim() == 1) and im.size(0) == 4: channels = 4
    elif isinstance(im, np.ndarray) and isinstance(im, np.floating):
        if   len(im.shape) == 1 and im.shape[0] == 1: channels = 1
        elif len(im.shape) == 2 or (len(im.shape) == 3 and im.shape[0] == 1): channels = 1
        elif (len(im.shape) == 3 or len(im.shape) == 1) and im.shape[0] == 2: channels = 2
        elif (len(im.shape) == 3 or len(im.shape) == 1) and im.shape[0] == 3: channels = 3
        elif (len(im.shape) == 3 or len(im.shape) == 0) and im.shape[0] == 4: channels = 4
    elif type(im) == ti.lang.matrix.Vector or type(im) == ti.lang.matrix.Matrix: 
        if   len(im) == 1: channels = 1
        elif len(im) == 2: channels = 2
        elif len(im) == 3: channels = 3
        elif len(im) == 4: channels = 4
    else: channels = 0
    return channels