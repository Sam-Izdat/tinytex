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
              - (NONE)
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

        INV_CUBIC_POLYNOMIAL        = 1<<10
        INV_QUARTIC_POLYNOMIAL      = 1<<11
        INV_QUADRATIC_RATIONAL      = 1<<12
        INV_CUBIC_RATIONAL          = 1<<13
        INV_RATIONAL                = 1<<14
        INV_PIECEWISE_QUADRATIC     = 1<<15
        INV_PIECEWISE_POLYNOMIAL    = 1<<16
        INV_TRIGONOMETRIC           = 1<<17

        T_POLYNOMIAL = CUBIC_POLYNOMIAL | QUARTIC_POLYNOMIAL | \
            QUINTIC_POLYNOMIAL | PIECEWISE_POLYNOMIAL | \
            INV_CUBIC_POLYNOMIAL | INV_QUARTIC_POLYNOMIAL | INV_PIECEWISE_POLYNOMIAL

        T_RATIONAL = QUADRATIC_RATIONAL | CUBIC_RATIONAL | RATIONAL | \
            INV_QUADRATIC_RATIONAL | INV_CUBIC_RATIONAL | INV_RATIONAL

        T_PIECEWISE = PIECEWISE_QUADRATIC | PIECEWISE_POLYNOMIAL | \
            INV_PIECEWISE_QUADRATIC | INV_PIECEWISE_POLYNOMIAL

        INVERSE = INV_CUBIC_POLYNOMIAL | INV_QUARTIC_POLYNOMIAL | \
            INV_QUADRATIC_RATIONAL | INV_CUBIC_RATIONAL | INV_RATIONAL | \
            INV_PIECEWISE_QUADRATIC | INV_PIECEWISE_POLYNOMIAL | INV_TRIGONOMETRIC

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

    @classmethod
    def apply(cls, 
        f:Union[Interpolant, str], 
        e0:float, 
        e1:float, 
        x:Union[float, torch.Tensor], 
        n:int=None) -> (float, torch.Tensor):
        """
        Smoothstep interpolation.

        :param f: Interpolant.
        :param e0: Lower edge (min).
        :param e1: Upper edge (max).
        :param x: Value.
        :param n: Order (if applicable).
        :rtype: float, torch.Tensor
        """
        if torch.is_tensor(x):
            x = torch.clamp((x - e0) / (e1 - e0), 0., 1.)
        else:
            x = min(max(((x - e0) / (e1 - e0), 0., 1.) , 0.), 1.)
        return cls.interpolate(f, x, n)

    @classmethod
    def interpolate(cls, f:Union[Interpolant, str], x:Union[float, torch.Tensor], n:int=None):
        interp = cls.Interpolant
        if type(f) == str: f = interp[f.strip().upper()]
        if f == interp.CUBIC_POLYNOMIAL:
            return cls.cubic_rational(x)
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
            return 0.5 - math.sin(math.asin(1. - 2. * x) / 3.)

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
            return math.sqrt(1. - math.sqrt(1. - x))

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
    def inv_quadratic_rational(x):
        """
        Inverse quadratic rational interpolation.
        """
        if torch.is_tensor(x):
            return (x - torch.sqrt(x * (1. - x))) / (2. * x - 1.)
        else:
            return (x - math.sqrt(x * (1. - x))) / (2. * x - 1.)

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
            a = math.pow(     x, 1. / 3.)
            b = math.pow(1. - x, 1. / 3.)
            return a / (a + b)

    @classmethod
    def rational(cls, x, n):
        """
        Rational interpolation.
        """
        if torch.is_tensor(x):
            torch.pow(x, n) / (torch.pow(x, n) + torch.pow(1. - x, n))
        else:
            math.pow(x, n) / (math.pow(x, n) + math.pow(1. - x, n))

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
            return 0.5 * math.pow(2. * x, n) if x < 0.5 else 1. - 0.5 * math.pow(2. * (1. - x), n)

    @classmethod
    def inv_piecewise_polynomial(cls, x, n):
        """
        Inverse piecewise polynomial interpolation.
        """
        if torch.is_tensor(x):
            torch.where(x < 0.5,  0.5 * torch.pow(2. * x, 1. / n), 1. - 0.5 * torch.pow(2. * (1. - x), 1. / n))
        else:
            0.5 * math.pow(2. * x, 1. / n) if x < 0.5 else 1. - 0.5 * math.pow(2. * (1. - x), 1. / n)

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