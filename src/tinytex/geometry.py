import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import typing
from typing import Union

from .util import *
from .resampling import Resampling

class Geometry:
    """
    Geometric surface processing. Where applicable, assumes a right-handed 
    y-up, x-right, z-back coordinate system and OpenGL-style normal maps.
    """

    err_size = "tensor must be sized [C, H, W] or [N, C, H, W]"
    err_height_ch = "height map tensor must have 1 channel"
    err_normal_ch = "normal map tensor must have 3 channels"
    err_angle_ch = "angle map must have 2 channels"

    @classmethod
    def normals_to_angles(cls,
        normal_map:torch.Tensor, 
        recompute_z:bool=False, 
        normalize:bool=False, 
        rescaled:bool=False):
        """
        Convert tangent-space normal vectors to scaled spherical coordinates.

        :param normal_map: Input normal map sized [N, C=3, H, W] or [C=3, H, W].
        :param recompute_z: Discard and recompute normals' z-channel before conversion.
        :param normalize: Normalize vectors before conversion.
        :param rescaled: Input and returned tensors should be in [0, 1] value range.
        :return: Scaled z-axis and y-axis angles tensor sized [N, C=2, H, W] or [C=2, H, W], 
            in range [0, 1].
        """
        ndim = len(normal_map.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: normal_map = normal_map.unsqueeze(0)
        assert normal_map.size(1), "normal map must have 3 channels"
        # Convert the RGB image tensor to a tensor with values in the range [-1, 1]
        
        if rescaled: normal_map = normal_map * 2. - 1.

        # Normalize the vector
        if recompute_z: normal_map = cls.recompute_z(normal_map, rescaled=False)
        if normalize: normal_map = cls.normalize(normal_map, rescaled=False)

        # Extract the red, green, and blue channels
        x = normal_map[:, 0:1, :, :]
        y = normal_map[:, 1:2, :, :]
        z = normal_map[:, 2:3, :, :]

        # Calculate angles and normalize [0, 1] range
        atan2_xz = torch.atan2(x, z) / (torch.pi * 0.5)
        acos_y = torch.acos(y) / torch.pi
        angles = torch.cat([atan2_xz, acos_y], dim=1)
        out = angles * 0.5 + 0.5 if rescaled else angles
        return out.squeeze(0) if nobatch else out

    @classmethod
    def angles_to_normals(cls, 
        angle_map:torch.Tensor, 
        recompute_z:bool=False, 
        normalize:bool=False, 
        rescaled:bool=False) -> torch.Tensor:
        """
        Convert scaled spherical coordinates to tangent-space normal vectors.

        :param angle_map: Scaled spherical coordinates tensor sized [N, C=2, H, W] or [C=2, H, W],
            in range [0, 1].
        :param reompute_z: Discard and recompute normal map's z-channel after conversion.
        :param normalize: Normalize vectors after conversion.
        :param rescaled: Input and returned tensors should be in [0, 1] value range.
        :return: Tensor of normals as unit vectors sized [N, C=3, H, W] or [C=3, H, W].
        """
        ndim = len(angle_map.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: angle_map = angle_map.unsqueeze(0)
        assert angle_map.size(1), cls.err_angle_ch
        if rescaled: angle_map = angle_map * 2. - 1.

        # Extract the z-angle and y-angle tensors
        z_angle_tensor = (angle_map[:, 0:1, :, :]) * (torch.pi * 0.5)
        y_angle_tensor = (angle_map[:, 1:2, :, :]) * torch.pi
        
        # Calculate the x, y, and z components of the normal map
        x_tensor = torch.sin(z_angle_tensor) * torch.sin(y_angle_tensor)
        y_tensor = torch.cos(y_angle_tensor)
        z_tensor = torch.cos(z_angle_tensor) * torch.sin(y_angle_tensor)
        
        # Stack the components into a single tensor and normalize the values
        normal_map_tensor = torch.cat((x_tensor, y_tensor, z_tensor), dim=1)
        if recompute_z: normal_map_tensor = cls.recompute_z(normal_map_tensor, rescaled=False)
        if normalize: normal_map_tensor = cls.normalize(normal_map_tensor, rescaled=False)
        out = normal_map_tensor * 0.5 + 0.5 if rescaled else normal_map_tensor
        return out.squeeze(0) if nobatch else out
        
    @classmethod
    def blend_normals(cls, 
        normals_base:torch.Tensor, 
        normals_detail:torch.Tensor, 
        rescaled:bool=False,
        eps:float=1e-8):
        """
        Blend two normal maps with reoriented normal map algorithm.
        
        :param normals_base: Base normals tensor sized [N, C=3, H, W] or [C=3, H, W] 
            as unit vectors of surface normals
        :param normals_detail: Detail normals tensor sized [N, C=3, H, W] or [C=3, H, W] 
            as unit vectors of surface normals
        :param rescaled: Input and returned unit vector tensors should be in [0, 1] value range.
        :param eps: epsilon
        :return: blended normals tensor sized [N, C=3, H, W] or [C=3, H, W] 
            as unit vectors of surface normals
        """
        assert normals_base.size() == normals_detail.size(), "base and detail tensors must have same number of dimensions"
        ndim = len(normals_base.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: 
            normals_base = normals_base.unsqueeze(0)
            normals_detail = normals_detail.unsqueeze(0)
        assert normals_base.size(1) == 3 and normals_detail.size(1) == 3, "inputs must have 3 channels"
        if rescaled:
            normals_base = normals_base * 2. - 1.
            normals_detail = normals_detail * 2. - 1.
        n1 = normals_base[:, :3, :, :]
        n2 = normals_detail[:, :3, :, :]
        
        a = 1 / (1 + n1[:, 2:3, :, :].clamp(-1 + eps, 1 - eps))
        b = -n1[:, 0:1, :, :] * n1[:, 1:2, :, :] * a
        
        # Basis
        b1 = torch.cat([1 - n1[:, 0:1, :, :] * n1[:, 0:1, :, :] * a, b, -n1[:, 0:1, :, :]], dim=1)
        b2 = torch.cat([b, 1 - n1[:, 1:2, :, :] * n1[:, 1:2, :, :] * a, -n1[:, 1:2, :, :]], dim=1)
        b3 = n1[:, :3, :, :]
        
        mask = (n1[:, 2:3, :, :] < -0.9999999).float()
        mask_ = 1 - mask
        
        # Handle the singularity
        b1 = b1 * mask_ + torch.cat([torch.zeros_like(mask), -torch.ones_like(mask), torch.zeros_like(mask)], dim=1) * mask
        b2 = b2 * mask_ + torch.cat([-torch.ones_like(mask), torch.zeros_like(mask), torch.zeros_like(mask)], dim=1) * mask
        
        # Rotate n2
        r = n2[:,0:1,:,:] * b1 + n2[:,1:2,:,:]*b2 + n2[:,2:3,:,:] * b3
        if rescaled: r = r * 0.5 + 0.5
        
        return r.squeeze(0) if nobatch else r

    @classmethod
    def height_to_normals(cls, height_map:torch.Tensor, rescaled:bool=False, eps:float=1e-4) -> torch.Tensor:
        """
        Compute tangent-space normals form height.

        :param height_map: Height map tensor sized [N, C=1, H, W] or [C=1, H, W] in [0, 1] range
        :param rescaled: Return unit vector tensor in [0, 1] value range.
        :param eps: Epsilon
        :return: normals tensor sized [N, C=3, H, W] or [C=3, H, W] as unit vectors of surface normals.
        """
        ndim = len(height_map.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: height_map = height_map.unsqueeze(0)
        assert height_map.size(1) == 1, cls.err_height_ch
        height_map = 1. - height_map
        device = height_map.device
        N = height_map.size(0)
        # height = 1 - height
        normals = []
        for i in range(N):
            dx = (torch.roll(height_map, -1, dims=3) - torch.roll(height_map, 1, dims=3))
            dy = (torch.roll(height_map, 1, dims=2) - torch.roll(height_map, -1, dims=2))
            z = torch.ones_like(dx)
            num = torch.cat([dx, dy, z], dim=1)
            denom = torch.sqrt(torch.sum(num ** 2, dim=1, keepdim=True) + eps)
            n = num / denom
            normals.append(n)

        normals = torch.cat(normals, dim=0)
        if rescaled: normals = normals * 0.5 + 0.5
        return normals.squeeze(0) if nobatch else normals


    @classmethod
    def normals_to_height(cls, 
        normal_map:torch.Tensor, 
        self_tiling:bool=False, 
        rescaled:bool=False,
        eps:float=torch.finfo(torch.float32).eps) -> (torch.Tensor, torch.Tensor):
        """
        Compute height from normals. Frankot-Chellappa algorithm.

        :param normal_map: Normal map tensor sized [N, C=3, H, W] or [C=3, H, W] 
            as unit vectors of surface normals.
        :param self_tiling: Treat surface as self-tiling.
        :param rescaled: Accept unit vector tensor in [0, 1] value range.
        :return: Height tensor sized [N, C=1, H, W] or [C=1, H, W] in [0, 1] range 
            and height scale tensor sized [N, C=1] or [C=1] in [0, inf] range.
        """
        ndim = len(normal_map.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: normal_map = normal_map.unsqueeze(0)
        assert normal_map.size(1) == 3, cls.err_normal_ch
        if rescaled: normal_map = normal_map * 2. - 1.
        device = normal_map.device
        N, _, H, W = normal_map.size()

        res_disp, res_scale = [], []
        for i in range(N):
            vec = normal_map[i]
            nx, ny = vec[0], vec[1]

            if not self_tiling:
                nxt = torch.cat([nx, -torch.flip(nx, dims=[1])], dim=1)
                nxb = torch.cat([torch.flip(nx, dims=[0]), -torch.flip(nx, dims=[0,1])], dim=1)
                nx = torch.cat([nxt, nxb], dim=0)

                nyt = torch.cat([ny, torch.flip(ny, dims=[1])], dim=1)
                nyb = torch.cat([-torch.flip(ny, dims=[0]), -torch.flip(ny, dims=[0,1])], dim=1)
                ny = torch.cat([nyt, nyb], dim=0)

            r, c = nx.shape
            rg = (torch.arange(r) - (r // 2 + 1)).float() / (r - r % 2)
            cg = (torch.arange(c) - (c // 2 + 1)).float() / (c - c % 2)

            u, v = torch.meshgrid(cg, rg, indexing='xy')
            u = torch.fft.ifftshift(u.to(device))
            v = torch.fft.ifftshift(v.to(device))
            gx = torch.fft.fft2(-nx)
            gy = torch.fft.fft2(ny)

            num = (-1j * u * gx) + (-1j * v * gy)
            denom = (u**2) + (v**2) + eps
            zf = num / denom
            zf[0, 0] = 0.0

            z = torch.real(torch.fft.ifft2(zf))
            disp, scale =  (z - torch.min(z)) / (torch.max(z) - torch.min(z)), float(torch.max(z) - torch.min(z))

            if not self_tiling: disp = disp[:H, :W]

            res_disp.append(disp.unsqueeze(0).unsqueeze(0))
            res_scale.append(torch.tensor(scale).unsqueeze(0))

        res_disp = torch.cat(res_disp, dim=0)
        res_scale = torch.cat(res_scale, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(res_disp.device)
        if nobatch:
            res_disp = res_disp.squeeze(0)
            res_scale = res_scale.squeeze(0)
        return res_disp, res_scale / 10.

    @classmethod
    def height_to_curvature(cls, 
        height_map:torch.Tensor, 
        blur_kernel_size:float=(1. / 128.), 
        blur_iter:int=1) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Compute mean curvature map from height map

        :param height_map: Height map tensor sized [N, C=1, H, W] or [C=1, H, W].
        :param blur_kernel_size: Size of blur kernel.
        :param blur_iter: Blur iterations.
        :return: curvature map tensor sized [N, C=1, H, W] in [0, 1] range,
            cavity map tensor [N, C=1, H, W] in [0, 1] range,
            peak map tensor [N, C=1, H, W] in [0, 1] range
        """
        # see: http://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation
        ndim = len(height_map.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: height_map = height_map.unsqueeze(0)
        assert height_map.size(1) == 1, cls.err_height_ch
        res_curv, res_cav, res_peak = [], [], []
        
        # Remap values to the range [0, 1]
        remap = lambda x : (x - torch.min(x)) / (torch.max(x) - torch.min(x))

        # Adjust blur kernel size based on image dimensions
        blur_kernel_size = int((H+W)/2 * blur_kernel_size) 
        if blur_kernel_size % 2 == 0: blur_kernel_size += 1

        for i in range(height_map.size(1)):
            hm = -height_map[i].clone()

            # First-order derivatives
            dz_dx = torch.gradient(hm, dim=1)[0]
            dz_dy = torch.gradient(hm, dim=2)[0]

            # Second-order derivatives
            d2z_dx2 = torch.gradient(dz_dx, dim=1)[0]
            d2z_dy2 = torch.gradient(dz_dy, dim=2)[0]
            d2z_dxdy = torch.gradient(dz_dx, dim=2)[0]

            # Construct Hessian matrix
            H = torch.stack([torch.stack([d2z_dx2, d2z_dxdy], dim=3), torch.stack([d2z_dxdy, d2z_dy2], dim=3)], dim=4)

            # Compute eigenvalues of Hessian matrix
            curvatures = torch.linalg.eigvalsh(H)

            # Remap eigenvalues to [0, 1]
            k1 = curvatures[..., 0]
            k2 = curvatures[..., 1]
            k1 = remap(k1)
            k2 = remap(k2)

            # Compute mean curvature
            mean_curv = (k1 + k2).unsqueeze(0) / 2

            # Apply iterative Gaussian blur
            for j in range(blur_iter): mean_curv = (mean_curv + TF.gaussian_blur(mean_curv, kernel_size=blur_kernel_size))/2

            # Compute concavity and peak maps
            res_curv.append(mean_curv)
            res_cav.append(1. - remap(mean_curv.clamp(0., 0.5)))
            res_peak.append(remap(mean_curv.clamp(0.5, 1.)))

        out = torch.cat(res_curv, dim=0), torch.cat(res_cav, dim=0), torch.cat(res_peak, dim=0)
        return out.squeeze(0) if nobatch else out

    @classmethod
    def compute_occlusion(cls, 
        normal_map:torch.Tensor=None, 
        height_map:torch.Tensor=None,
        height_scale:Union[torch.Tensor, float]=None, 
        radius:float=0.08, 
        samples:int=512,
        rescaled:bool=False) -> (torch.Tensor, torch.Tensor):
        """
        Compute ambient occlusion and bent normals from normal map and/or height map.

        :param height_map: Height map tensor sized [N, C=1, H, W] or [C=1, H, W] in [0, 1] range.
        :param normal_map: Normal map tensor sized [N, C=3, H, W] or [C=3, H, W] as unit vectors.
            of surface normals
        :param height_scale: Height scale as tensor sized [N, C=1] or [C=1], or as float.
        :param radius: Occlusion radius.
        :param samples: Number of occlusion samples per pixel.
        :param rescaled: Input and returned unit vector tensors should be in [0, 1] value range.
        :return: Ambient occlusion tensor sized [N, C=1, H, W] or [C=1, H, W], 
            bent normals tensor sized [N, C=3, H, W] or [C=3, H, W].
        """
        assert normal_map is not None or height_map is not None, "normal map and height map cannot both be None"
        if normal_map is None:
            normal_map = cls.height_to_normals(height_map)
        if height_map is None:
            height_map, height_scale = cls.normals_to_height(normal_map)
        assert height_map.size() == normal_map.size(), "height map and normal map tensors must have same number of dimensions"
        ndim = len(height_map.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: 
            height_map = height_map.unsqueeze(0)
            normal_map = normal_map.unsqueeze(0)
            height_scale = height_scale.unsqueeze(0) if torch.is_tensor(height_scale) else \
                torch.tensor(height_scale, dtype=height_map.dtype, device=height_map.device).unsqueeze(0)
        assert height_map.size(1) == 1, cls.err_height_ch
        assert normal_map.size(1) == 3, cls.err_normal_ch
        assert height_map.size()[2:4] == normal_map.size()[2:4], "height map and normal map must have same height and width"
        assert height_map.size(0) == normal_map.size(0), "height map and normal map must have same batch size"
        if rescaled: normal_map = normal_map * 2. - 1.

        device = height_map.device
        N, _, H, W = height_map.size()
        res_ao, res_bn = [], []
        with torch.no_grad():
            for i in range(N):
                hm, nm, hs = height_map[i], normal_map[i], height_scale[i]
                hm = hm * 2. * hs
                pos_nc = cls.__height_to_pos(hm, device=device)
                dir_nc = cls.__norm_to_dir(nm, normalize_ip=True)
                sample = torch.zeros_like(dir_nc)
                ao, bn = torch.zeros_like(hm), torch.zeros_like(nm)
                for j in range(samples):
                    dir_sample = cls.__cwhs(H*W, device=device)
                    dir_sample = F.normalize(torch.cat((
                        dir_nc[:, 0:1] + dir_sample[:, 0:1],
                        dir_nc[:, 1:2] + dir_sample[:, 1:2],
                        dir_nc[:, 2:3] * dir_sample[:, 2:3]
                        ), dim=1))
                    samples = pos_nc + radius * dir_sample
                    samples = samples.reshape(H, W, 3).permute(2,0,1)
                    grid = torch.stack((samples[0,:,:], samples[1,:,:]), dim=-1).unsqueeze(0)
                    height_at_sample = F.grid_sample(hm.unsqueeze(0), grid, padding_mode="reflection", align_corners=False)
                    mask = height_at_sample.squeeze(0)[0:1,:,:] > samples[2:3,:,:] 
                    mask_bn = mask.expand(3, -1, -1)
                    ao[mask] += 1
                    unoccluded_vec = dir_sample.reshape(H, W, 3).permute(2,0,1)
                    bn[~mask_bn] += unoccluded_vec[~mask_bn]
                ao = ao.float() / samples
                bn[bn == 0] = nm[bn == 0] # FIXME ~~~~~~~~~~~~~~~
                bn = F.normalize(bn, dim=0)
                bn = torch.cat((bn[0:1,:,:], -bn[1:2,:,:], bn[2:3,:,:]), dim=0)
                res_ao.append(1 - ao.unsqueeze(0))
                res_bn.append(bn.unsqueeze(0))
            res_ao = torch.cat(res_ao, dim=0)
            res_bn = torch.cat(res_bn, dim=0)
        if nobatch:
            res_ao = res_ao.squeeze(0)
            res_bn = res_bn.squeeze(0)
        if rescaled:
            res_bn = res_bn * 0.5 + 0.5
        return res_ao, res_bn

    @classmethod
    def normalize(cls, normal_map:torch.Tensor, rescaled:bool=False) -> torch.Tensor:
        """
        Normalize xyz vectors to unit length.

        :param normal_map: Tensor of normal vectors sized [N, C=3, H, W] or [C=3, H, W].
        :param rescaled: Input and returned unit vector tensors should be in [0, 1] value range.
        :return: Normalized tensor sized [N, C=3, H, W] or [C=3, H, W].
        """
        ndim = len(normal_map.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: normal_map = normal_map.unsqueeze(0)
        assert normal_map.size(1) == 3, cls.err_normal_ch
        if rescaled: normal_map = normal_map * 2. - 1.
        # normal_map = normal_map / torch.sqrt(torch.sum(normal_map ** 2, dim=1, keepdim=True))
        out = F.normalize(normal_map)
        if rescaled: out = out * 0.5 + 0.5
        return out.squeeze(0) if nobatch else out

    @classmethod
    def recompute_z(cls, normal_map:torch.Tensor, rescaled:bool=False) -> torch.Tensor:
        """
        Discard and recompute the z component of xyz vectors for a tangent-space normal map.

        :param normal_map: Tensor of normal vectors sized [N, C=3, H, W] or [C=3, H, W].
        :param rescaled: Input and returned unit vector tensors should be in [0, 1] value range.
        :return: Normals tensor with reconstructed z-channel sized [N, C=3, H, W] or [C=3, H, W].
        """
        ndim = len(normal_map.size())
        assert ndim == 3 or ndim == 4, cls.err_size
        nobatch = ndim == 3
        if nobatch: normal_map = normal_map.unsqueeze(0)
        if rescaled: normal_map = normal_map * 2. - 1.
        x = normal_map[:, 0, :, :]
        y = normal_map[:, 1, :, :]
        z = torch.sqrt(torch.clamp(1 - torch.pow(x.detach(), 2) - torch.pow(y.detach(), 2), min=0))
        normal_map[:, 2, :, :] = z.squeeze()
        if rescaled: normal_map = normal_map * 0.5 + 0.5
        return normal_map.squeeze(0) if nobatch else normal_map

    @classmethod
    def __cwhs(cls, n, device=torch.device('cpu')):
        """
        Generate n samples with cosine weighted hemisphere sampling

        :param int n: number of samples
        :param device: device for tensors (i.e. cpu or cuda)
        :return: tensor of n samples direction sized [N, 3]
        :rtype: torch.tensor
        """
        with torch.no_grad():
            u = torch.rand(n, device=device)
            v = torch.rand(n, device=device)
            phi = 2 * math.pi * u
            cos_theta = 1. - v
            sin_theta = torch.sqrt(1 - cos_theta ** 2)
            x = sin_theta * torch.cos(phi)
            y = sin_theta * torch.sin(phi)
            z = cos_theta
        return torch.stack([x, y, z], dim=1)

    @classmethod
    def __norm_to_dir(cls, normal_map:torch.Tensor, normalize_ip:bool=False):
        """
        Convert [C, H, W] sized image tensor of normal vectors to [N, C] tensor of direction vectors.

        :param torch.tensor normal_map: normal map tensor sized [C=3, H, W] as unit vectors
        :param bool normalize_ip: normalize input
        :return: direction tensor sized [H*W, C=3] 
        :rtype: torch.tensor
        """
        with torch.no_grad():
            C, H, W = normal_map.shape
            nt = normal_map
            nt = torch.cat((nt[0:1,:,:],-nt[1:2,:,:],nt[2:3,:,:]), dim=0)
            nt = nt.view(C, -1).T
            if normalize_ip: nt = nt / torch.linalg.vector_norm(nt, dim=1, keepdim=True)    
        return nt

    def __height_to_pos(height_map:torch.Tensor, device=torch.device('cpu')) -> torch.Tensor:
        """
        Convert image tensor of height values to flat tensor of position vectors.

        :param torch.tensor height_map: height map tensor sized [C=1, H, W] in [0, 1] range
        :param device: device for tensors (i.e. cpu or cuda)
        :return: position tensor sized [H*W, C=3] 
        :rtype: torch.tensor
        """
        with torch.no_grad():
            C, H, W = height_map.shape
            x = torch.linspace(-1, 1, W, device=device)
            y = torch.linspace(-1, 1, H, device=device)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            z = torch.ones_like(xx).to(device) * height_map.squeeze()
            pos_tensor = torch.stack([xx, yy, z, height_map.squeeze()], dim=0)
            pos_tensor = pos_tensor.reshape(4, -1).T[:, :3]
        return pos_tensor