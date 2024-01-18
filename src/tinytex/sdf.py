import imageio
import numpy as np
import matplotlib.pyplot as plt
import skfmm

import torch

from .util import *
from .resampler import Resampler
from .interpolation import Smoothstep

class SDF:
    @classmethod
    def compute(cls, im:torch.Tensor, tiling:bool=True, length_factor:float=0.1, threshold=0.) -> torch.Tensor:
        H, W = im.shape[1:]
        if threshold > 0.: im = (im > 0.5)
        if tiling: im = Resampler.tile(im, (H*3, W*3))
        im = im[0,...].numpy()
        im_norm = im * 2. - 1.
        lsq = (H*3)**2 + (W*3)**2 if tiling else H**2 + W**2
        max_dist = np.sqrt(lsq) * length_factor
        distance = (skfmm.distance(im_norm) * 0.5 + (max_dist * 0.5)) / max_dist
        if tiling: distance = distance[H:H*2,W:W*2]
        return torch.from_numpy(distance).unsqueeze(0)

    @classmethod
    def render(cls, 
        sdf:torch.Tensor, 
        shape=tuple, 
        edge0:float=0.496, 
        edge1:float=0.498, 
        value0:float=0., 
        value1:float=1.,
        interpolant='quintic_polynomial',
        mode:str='bilinear'):
        H, W = shape[0], shape[1]
        sdf = Resampler.resize(sdf, (H, W), mode=mode)
        ones = torch.ones_like(sdf)

        interp = Smoothstep.apply(interpolant, edge0, edge1, sdf)

        render = torch.lerp(ones * value0, ones * value1, interp)
        return render

# # print(im.shape)
# # exit()
# # ima = np.array(im)

# # binary_image = (im > 0.2).astype(np.uint8)



# foo = (skfmm.distance(binary_image) * 0.5 + (max_dist * 0.5)) / max_dist #/max_dist # / 50. * 0.5 + 0.5

# foo = foo[H:H*2,W:W*2]
# # print(foo.min(), foo.max())
# # exit()
# # foo = foo.astype(np.float32)


# binary_image = binary_image.astype(np.float32)
# fsio.save_image(torch.from_numpy(foo).unsqueeze(0), '../../out/a_sdf.png') 
# fsio.save_image(torch.from_numpy(binary_image).unsqueeze(0), '../../out/a_bin.png')


# def smooth_step(edge0, edge1, x):
#     # Clamp x to the range [0, 1]
#     x = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
#     # Evaluate the smooth step function
#     return x * x * (3 - 2 * x)

# bar = Resampler.resize(torch.from_numpy(foo).unsqueeze(0), (512, 512), mode='bicubic')
# ones = torch.ones_like(bar)
# zeros = torch.zeros_like(bar)
# # edge1 = ones * 0.6
# # edge2 = ones * 0.4
# # render = smooth_step(0.496, 0.512, bar)
# render = torch.lerp(ones * 0.85, ones * 0.15, smooth_step(0.439, 0.54, bar))
# # render = torch.where( (bar > 0.495) & (bar < 0.5), 0.75, render)
# fsio.save_image(render, '../../out/a_rendered.png') 

# print(foo.shape, foo.dtype)
# exit()


# # # Differentiate the inside / outside region
# # phi = np.int64(np.any(ima[:, :, :3], axis = 2))
# # # The array will go from - 1 to 0. Add 0.5(arbitrary) so there 's a 0 contour.
# # phi = np.where(phi, 0, -1) + 0.5
# # print(phi, phi.min(), phi.max())

# # dx = cell(pixel) 
# sd = skfmm.distance(phi, dx = 1)

# print(im.shape)
# print(ima.shape)
# print(phi.shape)
# print(sd.shape)
# exit()

# # # Show phi
# # plt.imshow(phi)
# # plt.xticks([])
# # plt.yticks([])
# # plt.colorbar()
# # plt.show()

# # # Compute signed distance
# # # dx = cell(pixel) size
# # sd = skfmm.distance(phi, dx = 1)

# # # Plot results
# # plt.imshow(sd)
# # plt.colorbar()
# # plt.show()