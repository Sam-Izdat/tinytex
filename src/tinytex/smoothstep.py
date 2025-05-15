import torch
import numpy as np

import typing
from typing import Union
from enum import IntEnum

class Smoothstep:
    """Smoothstep interpolation."""
    
    class Interpolant(IntEnum):
        """
        Interpolant. See: https://iquilezles.org/articles/smoothsteps/
        
        .. list-table:: Available Interpolants:
            :widths: 40 40 20
            :header-rows: 1

            * - Identifier
              - Inverse Identifier
              - Continuity
            * - CUBIC_POLYNOMIAL
              - INV_CUBIC_POLYNOMIAL
              - C1
            * - QUARTIC_POLYNOMIAL
              - INV_QUARTIC_POLYNOMIAL
              - C1
            * - QUINTIC_POLYNOMIAL
              - 
              - C2
            * - QUADRATIC_RATIONAL
              - INV_QUADRATIC_RATIONAL
              - C1
            * - CUBIC_RATIONAL
              - INV_CUBIC_RATIONAL
              - C2
            * - RATIONAL
              - INV_RATIONAL
              - CV
            * - PIECEWISE_QUADRATIC
              - INV_PIECEWISE_QUADRATIC
              - C1
            * - PIECEWISE_POLYNOMIAL
              - INV_PIECEWISE_POLYNOMIAL
              - CV
            * - TRIGONOMETRIC
              - INV_TRIGONOMETRIC
              - C1
        """
        CUBIC_POLYNOMIAL            = 1<<0
        QUARTIC_POLYNOMIAL          = 1<<1
        QUINTIC_POLYNOMIAL          = 1<<2
        QUADRATIC_RATIONAL          = 1<<3
        CUBIC_RATIONAL              = 1<<4
        RATIONAL                    = 1<<5
        PIECEWISE_QUADRATIC         = 1<<6
        PIECEWISE_POLYNOMIAL        = 1<<7
        TRIGONOMETRIC               = 1<<8

        INV_CUBIC_POLYNOMIAL        = 1<<0<<16
        INV_QUARTIC_POLYNOMIAL      = 1<<1<<16
        # No inverse quintic -----------------
        INV_QUADRATIC_RATIONAL      = 1<<3<<16
        INV_CUBIC_RATIONAL          = 1<<4<<16
        INV_RATIONAL                = 1<<5<<16
        INV_PIECEWISE_QUADRATIC     = 1<<6<<16
        INV_PIECEWISE_POLYNOMIAL    = 1<<7<<16
        INV_TRIGONOMETRIC           = 1<<8<<16

        T_POLYNOMIAL = CUBIC_POLYNOMIAL | QUARTIC_POLYNOMIAL | \
            QUINTIC_POLYNOMIAL | PIECEWISE_POLYNOMIAL | \
            INV_CUBIC_POLYNOMIAL | INV_QUARTIC_POLYNOMIAL | INV_PIECEWISE_POLYNOMIAL

        T_RATIONAL = QUADRATIC_RATIONAL | CUBIC_RATIONAL | RATIONAL | \
            INV_QUADRATIC_RATIONAL | INV_CUBIC_RATIONAL | INV_RATIONAL

        T_PIECEWISE = PIECEWISE_QUADRATIC | PIECEWISE_POLYNOMIAL | \
            INV_PIECEWISE_QUADRATIC | INV_PIECEWISE_POLYNOMIAL

        FORWARD = CUBIC_POLYNOMIAL | QUARTIC_POLYNOMIAL | \
            QUINTIC_POLYNOMIAL | QUADRATIC_RATIONAL | CUBIC_RATIONAL | RATIONAL | \
            PIECEWISE_QUADRATIC | PIECEWISE_POLYNOMIAL | TRIGONOMETRIC

        INVERSE = INV_CUBIC_POLYNOMIAL | INV_QUARTIC_POLYNOMIAL | \
            INV_QUADRATIC_RATIONAL | INV_CUBIC_RATIONAL | INV_RATIONAL | \
            INV_PIECEWISE_QUADRATIC | INV_PIECEWISE_POLYNOMIAL | INV_TRIGONOMETRIC

        HAS_ORDER = RATIONAL | INV_RATIONAL | PIECEWISE_POLYNOMIAL | INV_PIECEWISE_POLYNOMIAL

        # Continuity
        # C1 - second derivative does not evaluate to zero
        # C2 - second derivative does evaluate to zero
        # CV - variable; C(n-1)
        C1 = CUBIC_POLYNOMIAL | QUARTIC_POLYNOMIAL | QUADRATIC_RATIONAL | \
            PIECEWISE_QUADRATIC | TRIGONOMETRIC | \
            INV_CUBIC_POLYNOMIAL | INV_QUARTIC_POLYNOMIAL | \
            INV_QUARTIC_POLYNOMIAL | INV_PIECEWISE_QUADRATIC | INV_TRIGONOMETRIC
        C2 = QUINTIC_POLYNOMIAL | CUBIC_RATIONAL | INV_CUBIC_RATIONAL
        CV = PIECEWISE_POLYNOMIAL | RATIONAL | INV_PIECEWISE_POLYNOMIAL | INV_RATIONAL

    def __init__(self, interpolant: Union[str, Interpolant], n: int = None):
        """
        Initialize a Smoothstep interpolator.

        :param interpolant: Interpolant type (string or Interpolant).
        :param n: Order for variable-order interpolants (if applicable).
        """
        self.interpolant = self._parse_interpolant(interpolant)
        assert self.interpolant & self.Interpolant.FORWARD, "Must be initialized with forward interpolant"
        if self.interpolant & self.Interpolant.HAS_ORDER:
            assert not n is None, "Interpolant requires explicit order (n)"
        self.n = n

    @classmethod
    def _parse_interpolant(cls, f: Union[str, Interpolant]):
        """
        Convert a string or Interpolant into a validated Interpolant enum.

        :param f: Interpolant as string or enum.
        :return: Interpolant enum.
        """
        if type(f) is str: 
            try:
                f = cls.Interpolant[f.strip().upper()]
            except KeyError:
                raise AssertionError("Unrecognized interpolant")
        assert isinstance(f, cls.Interpolant), "Unrecognized interpolant"
        return f

    @classmethod
    def _invert_interpolant(cls, f: Interpolant):
        """
        Compute inverse interpolant enum value.

        :param f: Forward Interpolant enum.
        :return: Inverse Interpolant enum.
        """
        inv = 0
        try:
            inv = cls.Interpolant(f << 16)
        except ValueError:
            raise AssertionError("No valid inverse interpolant")
        assert inv & cls.Interpolant.INVERSE, "No valid inverse interpolant"
        return inv

    def forward(self, e0, e1, x):
        """
        Apply forward smoothstep interpolation.

        :param e0: Lower bound.
        :param e1: Upper bound.
        :param x: Input value(s).
        :return: Interpolated output.
        """
        return self.interpolate(self.interpolant, self._normalize(x, e0, e1), self.n)

    def inverse(self, y):
        """
        Apply inverse interpolation.

        :param y: Interpolated value.
        :return: Original input value.
        """
        return self.interpolate(self._invert_interpolant(self.interpolant), y, self.n)

    @classmethod
    def _normalize(cls, x:Union[float, torch.Tensor], e0:float = 0., e1:float = 1.):  
        """
        Normalize input value(s) to [0, 1] range.

        :param x: Input value(s).
        :param e0: Lower edge.
        :param e1: Upper edge.
        :return: Normalized value(s).
        """      
        if torch.is_tensor(x):
            return torch.clamp((x - e0) / (e1 - e0), 0., 1.)
        else:
            return min(max((x - e0) / (e1 - e0), 0.), 1.)

    @classmethod
    def apply(cls, 
        f:Union[Interpolant, str], 
        e0:float, 
        e1:float, 
        x:Union[float, torch.Tensor], 
        n:int=None) -> (float, torch.Tensor):
        """
        Static smoothstep evaluation with interpolant.

        :param f: Interpolant.
        :param e0: Lower edge (min).
        :param e1: Upper edge (max).
        :param x: Value.
        :param n: Order (if applicable).
        :return: Interpolated result.
        """
        f = cls._parse_interpolant(f)
        if f & cls.Interpolant.HAS_ORDER:
            assert not n is None, "Interpolant requires explicit order (n)"
        return cls.interpolate(f, cls._normalize(x, e0, e1), n)

    @classmethod
    def interpolate(cls, f:Union[Interpolant, str], x:Union[float, torch.Tensor], n:int=None):
        """
        Dispatch interpolation function based on interpolant.

        :param f: Interpolant.
        :param x: Normalized input.
        :param n: Order (if applicable).
        :return: Interpolated output.
        """
        interp = cls.Interpolant
        f = cls._parse_interpolant(f)
        if f == interp.CUBIC_POLYNOMIAL:
            return cls.cubic_polynomial(x)
        elif f == interp.INV_CUBIC_POLYNOMIAL:
            return cls.inv_cubic_polynomial(x)
        elif f == interp.QUARTIC_POLYNOMIAL:
            return cls.quartic_polynomial(x)
        elif f == interp.INV_QUARTIC_POLYNOMIAL:
            return cls.inv_quartic_polynomial(x)
        elif f == interp.QUINTIC_POLYNOMIAL:
            return cls.quintic_polynomial(x)
        elif f == interp.QUADRATIC_RATIONAL:
            return cls.quadratic_rational(x)
        elif f == interp.INV_QUADRATIC_RATIONAL:
            return cls.inv_quadratic_rational(x)
        elif f == interp.CUBIC_RATIONAL:
            return cls.cubic_rational(x)
        elif f == interp.INV_CUBIC_RATIONAL:
            return cls.inv_cubic_rational(x)
        elif f == interp.RATIONAL:
            return cls.rational(x, n)
        elif f == interp.INV_RATIONAL:
            return cls.inv_rational(x, n)
        elif f == interp.PIECEWISE_QUADRATIC:
            return cls.piecewise_quadratic(x)
        elif f == interp.INV_PIECEWISE_QUADRATIC:
            return cls.inv_piecewise_quadratic(x)
        elif f == interp.PIECEWISE_POLYNOMIAL:
            return cls.piecewise_polynomial(x, n)
        elif f == interp.INV_PIECEWISE_POLYNOMIAL:
            return cls.inv_piecewise_polynomial(x, n)
        elif f == interp.TRIGONOMETRIC:
            return cls.trigonometric(x)
        elif f == interp.INV_TRIGONOMETRIC:
            return cls.inv_trigonometric(x)
        else:
            raise ValueError('unrecognized interpolant')

    @classmethod
    def cubic_polynomial(cls, x):
        """
        Cubic polynomial - Hermite interpolation.
        """
        return x * x * (3. - 2. * x)

    @classmethod
    def inv_cubic_polynomial(cls, x):
        """
        Inverse cubic polynomial interpolation.
        """
        if torch.is_tensor(x):
            return 0.5 - torch.sin(torch.asin(1. - 2. * x) / 3.)
        else:
            return 0.5 - np.sin(np.arcsin(1. - 2. * x) / 3.)

    @classmethod
    def quartic_polynomial(cls, x):
        """
        Quartic polynomial interpolation.
        """
        return x * x * (2. - x * x)

    @classmethod
    def inv_quartic_polynomial(cls, x):
        """
        Inverse quartic polynomial interpolation.
        """
        if torch.is_tensor(x):
            return torch.sqrt(1. - torch.sqrt(1. - x))
        else:
            return np.sqrt(1. - np.sqrt(1. - x))

    @classmethod
    def quintic_polynomial(cls, x):
        """
        Quintic polynomial interpolation.
        """
        return x * x * x * (x * (x * 6. - 15.) + 10.)

    @classmethod
    def quadratic_rational(cls, x):
        """
        Quadratic rational interpolation.
        """
        return x * x / (2. * x * x - 2. * x + 1.)

    @classmethod
    def inv_quadratic_rational(cls, x):
        """
        Inverse quadratic rational interpolation.
        """
        if torch.is_tensor(x):
            return (x - torch.sqrt(x * (1. - x))) / (2. * x - 1.)
        else:
            return (x - np.sqrt(x * (1. - x))) / (2. * x - 1.)

    @classmethod
    def cubic_rational(cls, x):
        """
        Cubic rational interpolation.
        """
        return x * x * x / (3. * x * x - 3. * x + 1.)

    @classmethod
    def inv_cubic_rational(cls, x):
        """
        Inverse cubic rational interpolation.
        """
        if torch.is_tensor(x):
            a = torch.pow(     x, 1. / 3.)
            b = torch.pow(1. - x, 1. / 3.)
            return a / (a + b)
        else: 
            a = np.power(     x, 1. / 3.)
            b = np.power(1. - x, 1. / 3.)
            return a / (a + b)

    @classmethod
    def rational(cls, x, n):
        """
        Rational interpolation.
        """
        if torch.is_tensor(x):
            return torch.pow(x, n) / (torch.pow(x, n) + torch.pow(1. - x, n))
        else:
            return np.power(x, n) / (np.power(x, n) + np.power(1. - x, n))

    @classmethod
    def inv_rational(cls, x, n):
        """
        Inverse rational interpolation.
        """
        return cls.rational(x, 1. / n)

    @classmethod
    def piecewise_quadratic(cls, x):
        """
        Piecewise quadratic interpolation.
        """
        if torch.is_tensor(x):
            return torch.where(x < 0.5, 2. * x * x, 2. * x * (2. - x) - 1.)
        else:
            return 2. * x * x if x < 0.5 else 2. * x * (2. - x) - 1.

    @classmethod
    def inv_piecewise_quadratic(cls, x):
        """
        Inverse piecewise quadratic interpolation.
        """
        if torch.is_tensor(x):
            return torch.where(x < 0.5, torch.sqrt(0.5 * x), 1. - torch.sqrt(0.5 - 0.5 * x))
        else:
            return math.sqrt(0.5 * x) if x < 0.5 else 1. - math.sqrt(0.5 - 0.5 *x)

    @classmethod
    def piecewise_polynomial(cls, x, n):
        """
        Piecewise polynomial interpolation.
        """
        if torch.is_tensor(x):
            return torch.where(x < 0.5, 0.5 * torch.pow(2. * x, n), 1. - 0.5 * torch.pow(2. * (1. - x), n))
        else:
            return 0.5 * np.power(2. * x, n) if x < 0.5 else 1. - 0.5 * np.power(2. * (1. - x), n)

    @classmethod
    def inv_piecewise_polynomial(cls, x, n):
        """
        Inverse piecewise polynomial interpolation.
        """
        if torch.is_tensor(x):
            return torch.where(x < 0.5,  0.5 * torch.pow(2. * x, 1. / n), 1. - 0.5 * torch.pow(2. * (1. - x), 1. / n))
        else:
            return 0.5 * np.power(2. * x, 1. / n) if x < 0.5 else 1. - 0.5 * np.power(2. * (1. - x), 1. / n)

    @classmethod
    def trigonometric(cls, x):
        """
        Trigonometric interpolation.
        """
        if torch.is_tensor(x):
            return 0.5 - 0.5 * torch.cos(torch.pi * x)
        else:
            return 0.5 - 0.5 * math.cos(math.pi * x)

    @classmethod
    def inv_trigonometric(cls, x):
        """
        Inverse trigonometric interpolation.
        """
        if torch.is_tensor(x):
            return torch.acos(1. - 2. * x) / torch.pi
        else:
            return math.acos(1. - 2. * x) / math.pi