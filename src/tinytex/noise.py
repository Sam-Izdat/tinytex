import torch
import numpy as np
from scipy.spatial import cKDTree

from tinycio import MonoImage

from util import *

class Noise:

    @staticmethod
    def __interpolant_quintic(t):
        return t*t*t*(t*(t*6 - 15) + 10)

    @staticmethod
    def __interpolant_smoothstep(t):
        return t*t*(3-2*t)

    # From: https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py
    # Copyright (c) 2019 Pierre Vigier

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    @classmethod
    def __perlin_np(cls, shape, res, tileable=(True, True), interpolant='quintic'):
        """Generate a 2D numpy array of perlin noise.

        Args:
            shape: The shape of the generated array (tuple of two ints).
                This must be a multple of res.
            res: The number of periods of noise to generate along each
                axis (tuple of two ints). Note shape must be a multiple of
                res.
            tileable: If the noise should be tileable along each axis
                (tuple of two bools). Defaults to (False, False).
            interpolant: The interpolation function, defaults to
                t*t*t*(t*(t*6 - 15) + 10).

        Returns:
            A numpy array of shape shape with the generated noise.

        Raises:
            ValueError: If shape is not a multiple of res.
        """

        if interpolant == 'quintic':
            interpolant = cls.__interpolant_quintic
        elif interpolant == 'smoothstep':
            interpolant = cls.__interpolant_smoothstep
        else:
            raise Exception('unrecognized interpolant')

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
                 .transpose(1, 2, 0) % 1
        # Gradients
        angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        if tileable[0]:
            gradients[-1,:] = gradients[0,:]
        if tileable[1]:
            gradients[:,-1] = gradients[:,0]
        gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
        g00 = gradients[    :-d[0],    :-d[1]]
        g10 = gradients[d[0]:     ,    :-d[1]]
        g01 = gradients[    :-d[0],d[1]:     ]
        g11 = gradients[d[0]:     ,d[1]:     ]
        # Ramps
        n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
        n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
        n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
        # Interpolation
        t = interpolant(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)       

    @classmethod
    def perlin(cls, shape, density=5, tileable=(True, True), interpolant='quintic'):
        assert density > 0., "density cannot be 0"
        assert is_power_of_two(shape[0]) and is_power_of_two(shape[1]), "height and width must be power-of-two"
        res = (
            find_closest_divisor(shape[0], np.ceil(shape[0]/256.*density)), 
            find_closest_divisor(shape[1], np.ceil(shape[1]/256.*density)))
        out = cls.__perlin_np(shape, res, tileable, interpolant)
        return torch.from_numpy(np.expand_dims(out, 0).astype(np.float32)*0.5+0.5).clamp(0., 1.)

    # From: https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py
    # Copyright (c) 2019 Pierre Vigier

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    @classmethod
    def __fractal_np(cls, 
            shape, res, octaves=1, persistence=0.5,
            lacunarity=2, tileable=(True, True),
            interpolant='quintic'
    ):
        """Generate a 2D numpy array of fractal noise.

        Args:
            shape: The shape of the generated array (tuple of two ints).
                This must be a multiple of lacunarity**(octaves-1)*res.
            res: The number of periods of noise to generate along each
                axis (tuple of two ints). Note shape must be a multiple of
                (lacunarity**(octaves-1)*res).
            octaves: The number of octaves in the noise. Defaults to 1.
            persistence: The scaling factor between two octaves.
            lacunarity: The frequency factor between two octaves.
            tileable: If the noise should be tileable along each axis
                (tuple of two bools). Defaults to (False, False).
            interpolant: The, interpolation function, defaults to
                t*t*t*(t*(t*6 - 15) + 10).

        Returns:
            A numpy array of fractal noise and of shape shape generated by
            combining several octaves of perlin noise.

        Raises:
            ValueError: If shape is not a multiple of
                (lacunarity**(octaves-1)*res).
        """
        noise = np.zeros(shape)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * cls.__perlin_np(
                shape, (min(frequency*res[0], shape[0]), min(frequency*res[1], shape[1])), tileable, interpolant
            )
            frequency *= lacunarity
            amplitude *= persistence
        return noise

    @classmethod
    def fractal(cls, 
            shape, density, octaves=1, persistence=0.5,
            lacunarity=2, tileable=(True, True),
            interpolant='quintic'
    ):
        assert density > 0., "density cannot be 0"
        assert is_power_of_two(shape[0]) and is_power_of_two(shape[1]), "height and width must be power-of-two"
        res = (
            find_closest_divisor(shape[0], np.ceil(shape[0]/256.*density)), 
            find_closest_divisor(shape[1], np.ceil(shape[1]/256.*density)))
        out = cls.__fractal_np(shape, res, octaves, persistence, lacunarity, tileable, interpolant)
        return torch.from_numpy(np.expand_dims(out, 0).astype(np.float32)*0.5+0.5).clamp(0., 1.)

    @classmethod
    def __worley_np(cls, shape, density, tileable=(True, True)):
        height, width = shape[0], shape[1]
        points = []
        base = [[np.random.randint(0, height), np.random.randint(0, width)] for _ in range(density)]
        
        for h in range(3):
            if not tileable[0] and h != 1: continue
            for w in range(3):
                if not tileable[1] and w != 1: continue
                for v in range(density):
                    h_offset = h * height
                    w_offset = w * width
                    points.append([base[v][0] + h_offset, base[v][1] + w_offset])

        coord = np.dstack(np.mgrid[0:height*3, 0:width*3])
        tree = cKDTree(points)
        distances = tree.query(coord, workers=-1)[0].astype(np.float32)
        return distances[height:height*2, width:width*2]

    @classmethod
    def worley(cls, shape, density=5, intensity=1, tileable=(True, True)):
        assert density > 0., "density cannot be 0"
        assert is_power_of_two(shape[0]) and is_power_of_two(shape[1]), "height and width must be power-of-two"
        density *= 10
        intensity = 0.01 * intensity
        out = cls.__worley_np(shape, density, tileable)
        return torch.from_numpy(np.expand_dims(out*intensity, 0).astype(np.float32)).clamp(0., 1.)