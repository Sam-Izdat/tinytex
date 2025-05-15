import unittest
import torch
from tinytex import Smoothstep

err_tol_mean = 1e-5
err_tol_max = 1e-3
order = 3

def is_single_bit(val):
    return val != 0 and (val & (val - 1)) == 0

class TestSmoothstep(unittest.TestCase):

    def test_forward_inverse_float(self):
        s = Smoothstep('cubic_polynomial')
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = s.forward(0.0, 1.0, x)
            x2 = s.inverse(y)
            self.assertAlmostEqual(x, x2, places=4)

    def test_forward_inverse_tensor(self):
        s = Smoothstep('cubic_polynomial')
        x = torch.linspace(0, 1, 100)
        y = s.forward(0.0, 1.0, x)
        x2 = s.inverse(y)
        self.assertTrue(torch.allclose(x, x2, atol=err_tol_max))

    def test_forward_inverse_tensor_all(self):
        x = torch.linspace(0, 1, 100)
        for interp in Smoothstep.Interpolant:
            if is_single_bit(interp) and (interp & Smoothstep.Interpolant.FORWARD) \
                and ((interp << 16) & Smoothstep.Interpolant.INVERSE):
                try:
                    n = order if interp & Smoothstep.Interpolant.HAS_ORDER else None
                    s = Smoothstep(interp.name, n=n)
                    y = s.forward(0.0, 1.0, x)
                    x2 = s.inverse(y)
                    self.assertTrue(torch.allclose(x, x2, atol=err_tol_max), f"Failed on {interp.name}")
                    if torch.abs(x-x2).mean().item() > err_tol_mean:
                        raise AssertionError('mean error too high:', torch.abs(x-x2).mean().item())
                except AssertionError as e:
                    raise e

    def test_quintic_has_no_inverse(self):
        s = Smoothstep('quintic_polynomial')
        with self.assertRaises(AssertionError):
            _ = s.inverse(0.5)

    def test_rational_requires_order(self):
        with self.assertRaises(AssertionError):
            _ = Smoothstep('rational')

        s = Smoothstep('rational', n=3)
        val = s.forward(0.0, 1.0, 0.5)
        self.assertIsInstance(val, float)

    def test_tensor_input_supported(self):
        s = Smoothstep('cubic_polynomial')
        x = torch.tensor([0.2, 0.5, 0.8])
        y = s.forward(0.0, 1.0, x)
        self.assertIsInstance(y, torch.Tensor)

    def test_invalid_interpolant_raises(self):
        with self.assertRaises(AssertionError):
            _ = Smoothstep('nonexistent')

    def test_invert_non_invertible(self):
        with self.assertRaises(AssertionError):
            Smoothstep._invert_interpolant(Smoothstep.Interpolant.QUINTIC_POLYNOMIAL)

if __name__ == '__main__':
    unittest.main()