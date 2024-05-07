import typing
import math
from contextlib import nullcontext

import torch
import numpy as np

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from tinycio import Spectral, ColorSpace

from .util import progress_bar

class Diffraction:

    """Chromatic point spread function computation."""

    @classmethod
    def chromatic_psf(cls, 
        aperture:torch.Tensor, 
        lens_distance:float=1.8, 
        resize:int=512,
        aperture_blur:int=5,
        context=None) -> torch.Tensor:
        """Compute chromatic point spread function."""
        ndim = len(aperture.size())
        assert ndim == 2 or ndim == 3, "aperture tensor needs to be sized [H, W] or [C=1, H, W]"
        nochan = ndim == 2
        if nochan: aperture = aperture.unsqueeze(0)
        assert aperture.size(0) == 1, "aperture tensor be monochromatic"
        aperture = cls.__prep_aperture(aperture, aperture_blur_size=aperture_blur)
        cm_tab = Spectral.cm_table()  
        max_dim_x = int(aperture.size(1) * (380. + (770. - 380.)) / 575.)
        max_dim_y = int(aperture.size(2) * (380. + (770. - 380.)) / 575.)
        psf_chr = []
        for i in range(3):
            psf_chr.append(torch.zeros((1, max_dim_y, max_dim_x), dtype=torch.cfloat))

        K = 1/(575*lens_distance)**2

        __ctx = context if context is not None and callable(context) else nullcontext
        with __ctx() as ctx:
            cb_callable = hasattr(ctx, 'update_status') and callable(ctx.update_status)
            cb = ctx.update_status if cb_callable else lambda a, b, c, d: None
            for i in range(cm_tab.size(0)):
                idx = (cm_tab.size(0)-1) - i
                # idx=i
                fac = (380. + idx * (770. - 380.) / cm_tab.size(0)) / 575.
                scale_x = int(float(aperture.size(1)) * fac) 
                scale_y = int(float(aperture.size(2)) * fac)

                cb(i, 1, cm_tab.size(0)-1, f"WL {int(380. + i * (770. - 380.) / cm_tab.size(0))}")

                ap = aperture.clone()    
                ap = F.interpolate(ap.unsqueeze(0), size=(scale_y, scale_x), mode="bilinear").squeeze(0)

                padding = T.Pad([
                    math.floor((max_dim_x-scale_x)/2), math.floor((max_dim_y-scale_y)/2),
                    math.ceil((max_dim_x-scale_x)/2), math.ceil((max_dim_y-scale_y)/2)
                ]) 
                ap = padding(ap)

                psf_chr[0][0:1,...] += torch.fft.ifftshift(torch.fft.ifftn(
                    K*torch.abs(torch.fft.fftn(torch.fft.fftshift(ap)))**2) * cm_tab[i,:][0])
                psf_chr[1][0:1,...] += torch.fft.ifftshift(torch.fft.ifftn(
                    K*torch.abs(torch.fft.fftn(torch.fft.fftshift(ap)))**2) * cm_tab[i,:][1])
                psf_chr[2][0:1,...] += torch.fft.ifftshift(torch.fft.ifftn(
                    K*torch.abs(torch.fft.fftn(torch.fft.fftshift(ap)))**2) * cm_tab[i,:][2])
            res = torch.cat(psf_chr, dim=0)
            res = F.interpolate(res.real.unsqueeze(0), size=(resize, resize), mode="bilinear").squeeze(0)
            return res

    @classmethod
    def render_glare(cls, 
        im:torch.Tensor, 
        psf:torch.Tensor, 
        color_space:ColorSpace.Variant, # color space of im
        aperture_radius:float, # assumed to be in cm
        diffraction_strength:float, # artist-controlled dial of diffraction strength; rec. 0 - 1
        lum_threshold:float=10.
        ):
        """Apply chromatic glare to using point spread function."""
        with torch.no_grad():            
            _, H_in, W_in = im.size()
            im_ten = im.clone()

            if H_in > W_in:
                pad_h = int(H_in//2)
                pad_w = ((H_in + pad_h*2) - W_in)//2
            else:
                pad_w = int(W_in//2)
                pad_h = ((W_in + pad_w*2) - H_in)//2

            padding = T.Pad([pad_w, pad_h])
            im_ten = padding(im_ten)

            _, H_padded, W_padded = im_ten.size()

            # I think the idea was using Rayleigh Criterion for scaling, assuming that aperture radius is in cm?
            # Then theta, in radians, is scaled by another user-defined factor.
            rayleigh_fac = (1.22 * 550.)/ (aperture_radius * 2. * 1e+7) 
            rayleigh_fac *= diffraction_strength * 1000.

            lum = ColorSpace.convert(im_ten, color_space, ColorSpace.Variant.CIE_XYY)[2] * rayleigh_fac
            lum_mask = (lum > lum_threshold).repeat(3,1,1)
            im_ten[~lum_mask] = 0.

            aspect = W_in/H_in
            im_min_dim = min(H_in, W_in)
            im_max_dim = min(H_padded, W_padded)

            psfl = []
            psfl.append(psf[0:1,:,:].clone())
            psfl.append(psf[1:2,:,:].clone())
            psfl.append(psf[2:3,:,:].clone())
            psfl[0] = F.interpolate(psfl[0].real.unsqueeze(0), size=[int((im_max_dim) ), int(im_max_dim)], mode="bilinear").squeeze(0)
            psfl[1] = F.interpolate(psfl[1].real.unsqueeze(0), size=[int((im_max_dim) ), int(im_max_dim)], mode="bilinear").squeeze(0)
            psfl[2] = F.interpolate(psfl[2].real.unsqueeze(0), size=[int((im_max_dim) ), int(im_max_dim)], mode="bilinear").squeeze(0)
            psfl[0] = T.functional.center_crop(psfl[0], (H_padded, W_padded))
            psfl[1] = T.functional.center_crop(psfl[1], (H_padded, W_padded))
            psfl[2] = T.functional.center_crop(psfl[2], (H_padded, W_padded))
            psfl[0] = torch.fft.fftshift(psfl[0])
            psfl[1] = torch.fft.fftshift(psfl[1])
            psfl[2] = torch.fft.fftshift(psfl[2])

            im_ten_xyz = ColorSpace.convert(im_ten, color_space, ColorSpace.Variant.CIE_XYZ) * rayleigh_fac
            
            im_ten_x = torch.fft.fftn(torch.fft.fftshift(im_ten_xyz[0:1,:,:].clone()))
            im_ten_y = torch.fft.fftn(torch.fft.fftshift(im_ten_xyz[1:2,:,:].clone()))
            im_ten_z = torch.fft.fftn(torch.fft.fftshift(im_ten_xyz[2:3,:,:].clone()))

            im_ten_x = torch.mul(im_ten_x, psfl[0][0:1,:,:])
            im_ten_y = torch.mul(im_ten_y, psfl[1][0:1,:,:])
            im_ten_z = torch.mul(im_ten_z, psfl[2][0:1,:,:])

            im_ten_x = torch.fft.ifftshift(torch.fft.ifftn(im_ten_x)) 
            im_ten_y = torch.fft.ifftshift(torch.fft.ifftn(im_ten_y))
            im_ten_z = torch.fft.ifftshift(torch.fft.ifftn(im_ten_z))

            out_ten = torch.cat([im_ten_x, im_ten_y, im_ten_z], dim=0)
            out_ten = T.functional.center_crop(out_ten, (H_in, W_in))
            out_ten = ColorSpace.convert(out_ten.real, ColorSpace.Variant.CIE_XYZ, ColorSpace.Variant.LMS)

            contrib = T.functional.center_crop(im_ten, (H_in, W_in))
            glare = T.functional.center_crop(out_ten.real.clamp(0.,torch.inf), (H_in, W_in))
        return glare, contrib

    @classmethod
    def __prep_aperture(cls, aperture_map:torch.Tensor, aperture_blur_size:int=0):
        """Pad and blur aperture map"""
        padding = T.Pad([aperture_map.size(2)//2, aperture_map.size(1)//2])
        out = padding(aperture_map.clone())

        if aperture_blur_size > 0:
            out = TF.gaussian_blur(
                out.unsqueeze(0), 
                kernel_size=aperture_blur_size).squeeze(0)
        return out