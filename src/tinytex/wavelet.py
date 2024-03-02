import torch

class Wavelet:

    """Wavelet transforms and coefficient culling."""

    # See Emil Mikulic's implementation: https://unix4lyfe.org/haar/

    scale = torch.sqrt(torch.tensor(2.0))

    @classmethod
    def haar(cls, a:torch.Tensor) -> torch.Tensor:
        """
        1D Haar transform.
        """
        l = a.size(1) 
        if l == 1: return a.clone()
        assert l % 2 == 0, f"length needs to be even"
        mid = (a[:, 0::2] + a[:, 1::2]) / cls.scale
        side = (a[:, 0::2] - a[:, 1::2]) / cls.scale
        return torch.cat((cls.haar(mid), side), dim=1)

    @classmethod
    def inverse_haar(cls, a:torch.Tensor) -> torch.Tensor:
        """
        1D inverse Haar transform.
        """
        l = a.size(1) 
        if l == 1: return a.clone()
        assert l % 2 == 0, "length needs to be even"
        mid = cls.inverse_haar(a[:, 0:l//2]) * cls.scale
        side = a[:, l//2:] * cls.scale
        out = torch.zeros(a.size(), dtype=torch.float32)
        out[:, 0::2] = (mid + side) / 2.
        out[:, 1::2] = (mid - side) / 2.
        return out

    @classmethod
    def haar_2d(cls, im:torch.Tensor) -> torch.Tensor:
        """
        2D Haar transform.
        """
        ndim = len(im.size())
        assert ndim == 2 or ndim == 3, "tensor must be sized [H, W] or [N, H, W]"
        nobatch = ndim == 2
        if nobatch: im = im.unsqueeze(0)
        _, H, W = im.shape
        rows = torch.zeros(im.shape, dtype=torch.float32)
        for x in range(H): rows[:, x] = cls.haar(im[:, x])
        cols = torch.zeros(im.shape, dtype=torch.float32)
        for y in range(W): cols[:, :, y] = cls.haar(rows[:, :, y])
        return cols.squeeze(0) if nobatch else cols

    @classmethod
    def inverse_haar_2d(cls, coeffs:torch.Tensor) -> torch.Tensor:
        """
        2D inverse Haar transform.
        """
        ndim = len(coeffs.size())
        assert ndim == 2 or ndim == 3, "tensor must be sized [H, W] or [N, H, W]"
        nobatch = ndim == 2
        if nobatch: coeffs = coeffs.unsqueeze(0)
        H, W = coeffs.shape[1:]
        cols = torch.zeros(coeffs.shape, dtype=torch.float32)
        for y in range(W): cols[:, :, y] = cls.inverse_haar(coeffs[:, :, y])
        rows = torch.zeros(coeffs.shape, dtype=torch.float32)
        for x in range(H): rows[:, x] = cls.inverse_haar(cols[:, x])
        return rows.squeeze(0) if nobatch else rows

    @classmethod
    def cull_haar_magnitude(cls, a:torch.Tensor, ratio:float) -> torch.Tensor:
        """
        Keep only the strongest Haar coefficients by magnitude.

        :param a: Haar wavelet coefficients tensor sized [F] or [N, F], where N is batch count 
        and F is coefficient count.
        :param ratio: Ratio of coefficients to cull, in range [0, 1].
        """

        ndim = len(a.size())
        assert ndim == 1 or ndim == 2, "tensor must be sized [F] or [N, F]"
        nobatch = ndim == 1
        if nobatch: a = a.unsqueeze(0)

        k_value = int((a.size(-1) - 1) * ratio)

        a_abs = a.abs()
        sorted_values, _ = torch.sort(a_abs, dim=-1)
        threshold_values = sorted_values[:, k_value]

        a[a_abs < threshold_values.unsqueeze(-1)] = 0.
        return a.squeeze(0) if nobatch else a

    @classmethod
    def cull_haar_2d_aw(cls, a:torch.Tensor, ratio:float) -> torch.Tensor:
        """
        Keep only the strongest Haar coefficients in a 2D image by area-weighted magnitude.

        :param a: 2D Haar wavelet coefficients tensor sized [H, W] or [N, H, W], where N is batch count, 
        H is height and W is width.
        :param ratio: Ratio of coefficients to cull, in range [0, 1].
        """
        ndim = len(a.size())
        assert ndim == 2 or ndim == 3, "tensor must be sized [H, W] or [N, H, W]"
        nobatch = ndim == 2
        if nobatch: a = a.unsqueeze(0)

        N, H, W = a.shape
        a_abs = a.clone()

        indices_x, indices_y = torch.meshgrid(torch.arange(a.size(1)), torch.arange(a.size(2)), indexing='ij')

        level = torch.ceil(torch.log(1 + torch.max(indices_x, indices_y))) 
        area = torch.pow(4, (H.bit_length() - level)).unsqueeze(0)
        a_abs = (a_abs.abs() * area.float()).view(a_abs.size(0), -1)

        sorted_values, _ = torch.sort(a_abs, dim=-1)
        k_value = int(a_abs.size(-1) * ratio)
        threshold_values_k = sorted_values[:, k_value]
        a = a.view(a.size(0), -1)
        a[a_abs < threshold_values_k.unsqueeze(-1)] = 0.
        a = a.reshape(N, H, W)
        return a.squeeze(0) if nobatch else a

    @classmethod
    def haar_bipolar(cls, im:torch.Tensor) -> torch.Tensor:
        """
        Scales Haar coefficients to range [0, 1]. 
        Returns [C=3, H, W] sized tensor where negative values are red, 
        positive values are blue, and zero is black.
        """
        n, h, w = im.shape
        im = im.clone()
        im /= torch.abs(im).max()
        out = torch.zeros((n, h, w, 3), dtype=torch.float32)
        a = 0.005
        b = 1. - a
        c = 0.5
        out[:, :, :, 0] = torch.where(im < 0, a + b * torch.pow(torch.abs(im / (im.min() - 0.001)), c), torch.tensor(0.0))
        out[:, :, :, 2] = torch.where(im > 0, a + b * torch.pow(torch.abs(im / (im.max() + 0.001)), c), torch.tensor(0.0))
        return out.permute(2, 0, 1)