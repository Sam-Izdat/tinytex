import torch

class Wavelet:

    # See: https://unix4lyfe.org/haar/
    # Based on Emil Mikulic's implementation.

    scale = torch.sqrt(torch.tensor(2.0))

    @classmethod
    def haar(cls, a:torch.Tensor) -> torch.Tensor:
        """
        1D Haar transform.
        """
        l = a.size(0) 
        if l == 1: return a.clone()
        assert l % 2 == 0, "length needs to be even"
        mid = (a[0::2] + a[1::2]) / cls.scale
        side = (a[0::2] - a[1::2]) / cls.scale
        return torch.cat((cls.haar(mid), side), dim=0)

    @classmethod
    def ihaar(cls, a:torch.Tensor) -> torch.Tensor:
        """
        1D inverse Haar transform.
        """
        l = a.size(0) 
        if l == 1: return a.clone()
        assert l % 2 == 0, "length needs to be even"
        mid = cls.ihaar(a[0:l//2]) * cls.scale
        side = a[l//2:] * cls.scale
        out = torch.zeros(a.size(0), dtype=torch.float32)
        out[0::2] = (mid + side) / 2.
        out[1::2] = (mid - side) / 2.
        return out

    @classmethod
    def haar_2d(cls, im:torch.Tensor) -> torch.Tensor:
        """
        2D Haar transform.
        """
        h, w = im.shape
        rows = torch.zeros(im.shape, dtype=torch.float32)
        for y in range(h): rows[y] = cls.haar(im[y])
        cols = torch.zeros(im.shape, dtype=torch.float32)
        for x in range(w): cols[:, x] = cls.haar(rows[:, x])
        return cols

    @classmethod
    def ihaar_2d(cls, coeffs:torch.Tensor) -> torch.Tensor:
        """
        2D inverse Haar transform.
        """
        h, w = coeffs.shape
        cols = torch.zeros(coeffs.shape, dtype=torch.float32)
        for x in range(w): cols[:, x] = cls.ihaar(coeffs[:, x])
        rows = torch.zeros(coeffs.shape, dtype=torch.float32)
        for y in range(h): rows[y] = cls.ihaar(cols[y])
        return rows

    @classmethod
    def strong_coeffs(cls, a:torch.Tensor, ratio:float) -> torch.Tensor:
        """
        Keep only the strongest values.
        """
        magnitude = sorted(torch.abs(a.flatten()))
        idx = int((len(magnitude) - 1) * (1. - ratio))
        return torch.where(torch.abs(a) > magnitude[idx], a, torch.tensor(0, dtype=a.dtype))

    @classmethod
    def haar_bipolar(cls, im:torch.Tensor) -> torch.Tensor:
        """
        Scales Haar coefficients to range [0, 1]. 
        Returns [C=3, H, W] sized tensor where negative values are red, 
        positive values are blue, and zero is black.
        """
        h, w = im.shape
        im = im.clone()
        im /= torch.abs(im).max()
        out = torch.zeros((h, w, 3), dtype=torch.float32)
        a = 0.005
        b = 1. - a
        c = 0.5
        out[:, :, 0] = torch.where(im < 0, a + b * torch.pow(torch.abs(im / (im.min() - 0.001)), c), torch.tensor(0.0))
        out[:, :, 2] = torch.where(im > 0, a + b * torch.pow(torch.abs(im / (im.max() + 0.001)), c), torch.tensor(0.0))
        return out.permute(2, 0, 1)