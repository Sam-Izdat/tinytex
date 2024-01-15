import numpy as np
import torch

# from perlin import generate_perlin_noise_2d, generate_fractal_noise_2d
# from worley import worley

from noise import Noise

from tinycio import fsio

from util import *
from resampler import Resampler
from tiler import Tiler
from atlas import Atlas

# Example usage:
# Assuming textures is a list of TextureRect objects








textures = []
rng = np.random.random

import os
directory = '../../out/atlas'
 
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    ext = os.path.splitext(filename)[1]
    if os.path.isfile(f) and (ext == '.png' or ext == '.jpg'):
    	print(filename)
    	im = fsio.truncate_image(fsio.load_image(f))
    	textures.append(im)


for i in range(160):
	rect = torch.ones(3, int(rng()*60), int(rng()*100))*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]])
	textures.append(rect)
textures.append(torch.ones(3, 250, 250)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 230, 230)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 240, 240)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 200, 200)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 150, 150)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 80, 80)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 60, 60)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))
textures.append(torch.ones(3, 50, 50)*torch.Tensor([[[rng()]], [[rng()]], [[rng()]]]))


# Pack textures
atl2 = Atlas()
atl = Atlas(force_auto_square=True)

atlas, index = atl.pack(textures)
fsio.save_image(atlas, '../../out/atlas.png')

atlas, index = Atlas.pack(textures)
fsio.save_image(atlas, '../../out/atlas2.png')
print(atlas.size(), index)


exit()



# np.random.seed(0)
# seed_everything(0)

# beef = fsio.load_image('../../out/beef.png')
# beef, r, c = Tiler.split(beef, 256)
# beef = beef[...,0:210,0:210]
# print(beef.size())
# beef = Tiler.blend(beef, r, c)
# beef = Tiler.merge(beef, r, c).squeeze(0)
# fsio.save_image(beef, '../../out/beef_blended.png')
# exit()


seed_everything(0)
height, width = 256, 256
# foo = Noise.worley((height, width), density=5, intensity=1)
foo = Noise.fractal((height, width), density=6, tileable=(True,True), octaves=5, persistence=0.5, lacunarity=2, interpolant='quintic')
foo = foo.roll([70, 70], dims=[1,2])
foo = Resampler.tile(foo, (512,512))
bar, r, c = Tiler.split(foo, 256)
# print(bar.size())
# bar = Tiler.blend(bar, r, c)
bar = Tiler.merge(bar, r, c)

# bar = Tiler.blend(foo.unsqueeze(0)).squeeze(0)
print(bar.size())
fsio.save_image(bar, '../../out/rtex2.png')
exit()






noise = Noise.fractal((height, width), density=6, tileable=(True,True), octaves=5, persistence=0.5, lacunarity=2, interpolant='quintic')
# noise = Noise.perlin((height, width), density=5, tileable=(True,True))
# noise = Resampler.tile(noise, (700,700))
noise = Resampler.tile_to_square(noise, 1024)
fsio.save_image(noise, '../../out/rtex.png')