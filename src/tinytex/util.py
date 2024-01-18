import torch
import numpy as np
import random

def find_closest_divisor(n:int, m:int) -> int: 
    """
    Find the divisor of n closest to m

    :return: Closest divisor.    
    """
    if n % m == 0: return m.astype(int)
    divisors = np.array([ i for i in range(1, n+1) if n % i == 0 ])
    divisions = n / divisors 
    return divisions[np.argmin(np.abs(m - divisions))].astype(int)

def is_pot(x:int) -> int:
    """
    Check if value is power-of-two.

    :return: True if power-of-two False if not.
    """
    return x > 0 and ((x & (x - 1)) == 0)

def next_pot(x:int) -> int:
    """
    Get next power-of-two value.

    :return: Next power-of-two integer.
    """
    return 2 ** np.ceil(np.log2(x))

# https://kaba.hilvi.org/pastel-1.6.0/pastel/math/interpolation/smoothstep.h.htm

# cubic
def smoothstep_cubic(edge0:float, edge1:float, x:torch.Tensor) -> torch.Tensor:
    """
    Cubic/Hermite interpolation.

    :return: Interpolated value.
    """
    x = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3 - 2 * x)

# quartic
def smoothstep_quartic(edge0:float, edge1:float, x:torch.Tensor) -> torch.Tensor:
    """
    Quartic interpolation.

    :return: Interpolated value.
    """
    x = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (2 - x * x)

# quintic
def smoothstep_quintic(edge0:float, edge1:float, x:torch.Tensor) -> torch.Tensor:
    """
    Quintic interpolation.

    :return: Interpolated value.
    """
    x = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)

def seed_everything(seed: int) -> bool:
    """
    Seed Python, NumPy and PyTorch RNGs.

    :return: True if successful.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return True