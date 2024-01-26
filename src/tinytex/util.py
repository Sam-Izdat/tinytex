import torch
import numpy as np
import random

from tqdm import tqdm

def closest_divisor(n:int, m:int) -> int: 
    """
    Find the divisor of n closest to m.

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

class _ProgressBar(tqdm):
    """Provides `update_status(n)` which uses `tqdm.update(delta_n)`."""
    def update_status(self, batches_done=1, steps_per_batch=1, steps_total=None, desc=''):
        if steps_total is not None: self.total = steps_total
        self.set_description(desc)
        return self.update(batches_done * steps_per_batch - self.n)

def progress_bar(desc:str="Computing"): 
    """Context to display a progressbar with tqdm."""
    return _ProgressBar(unit=' steps', unit_scale=True, unit_divisor=1024, miniters=1, desc=desc)