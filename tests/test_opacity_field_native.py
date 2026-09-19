"""Per-vertex opacity field: native rasterizer contract.

With ``opacity_field`` a face's opacity at a pixel is its three vertex
opacities interpolated with the barycentrics that already interpolate colour,
instead of the min over them. These tests pin what the model rests on:

1. a face whose corners agree renders as it does under the min;
2. distinct corners make opacity vary across the face;
3. the backward is the exact derivative of that forward, checked against
   finite differences, and every corner receives it;
4. the published min path is untouched when the field is off.

Needs the rebuilt extension and a CUDA device, so it skips elsewhere.
"""

import math
import unittest

import torch

try:
    from diff_triangle_rasterization import (
        TriangleRasterizationSettings,
        TriangleRasterizer,
    )
    from utils.graphics_utils import getProjectionMatrix
    RASTERIZER_AVAILABLE = True
except Exception:  # noqa: BLE001 - native extension absent off the A40
    RASTERIZER_AVAILABLE = False

FIELD_SUPPORTED = RASTERIZER_AVAILABLE and "opacity_field" in getattr(
    TriangleRasterizationSettings, "_fields", ()
)

SIZE = 64
# Distinct and inside the trained range; the largest stays below the backward's
# 0.99 alpha clamp so the image is linear in every corner opacity.
OPACITIES = (0.81, 0.88, 0.97)
VERTICES = ((-0.62, -0.52, 2.0), (0.62, -0.52, 2.0), (0.0, 0.64, 2.0))
COLORS = ((0.20, 0.34, 0.42), (0.43, 0.18, 0.12), (0.11, 0.51, 0.25))


def _settings(device, field):
    field_of_view = 1.0
    return TriangleRasterizationSettings(
        image_height=SIZE,
        image_width=SIZE,
        tanfovx=math.tan(field_of_view / 2),
        tanfovy=math.tan(field_of_view / 2),
        bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, dtype=torch.float32, device=device),
        projmatrix=getProjectionMatrix(0.01, 100.0, field_of_view, field_of_view)
        .transpose(0, 1)
        .to(device),
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device=device),
        prefiltered=False,
        debug=False,
        opacity_field=field,
    )


def _render(device, field, opacities, grad=False):
    weights = torch.tensor(opacities, dtype=torch.float32, device=device)
    weights.requires_grad_(grad)
    image = TriangleRasterizer(_settings(device, field))(
        vertices=torch.tensor(VERTICES, dtype=torch.float32, device=device),
        triangles_indices=torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device),
        vertex_weights=weights,
        sigma=1.0,
        scaling=torch.zeros(1, dtype=torch.float32, device=device),
        colors_precomp=torch.tensor(COLORS, dtype=torch.float32, device=device),
    )[0]
    return image, weights


def _loss(image):
    # A fixed, asymmetric linear functional: the image is linear in each corner
    # opacity, so central differences of this loss are exact up to rounding.
    probe = torch.linspace(-1.0, 1.0, image.numel(), device=image.device)
    return (image.flatten() * probe).sum()


@unittest.skipUnless(
    FIELD_SUPPORTED and torch.cuda.is_available(),
    "needs a CUDA device and a rasterizer rebuilt with opacity_field",
)
class OpacityFieldNativeTest(unittest.TestCase):
    DEVICE = "cuda"

    def test_uniform_corners_match_the_min(self):
        """With equal corners the field and the min describe the same face."""
        uniform = (0.86, 0.86, 0.86)
        pooled, _ = _render(self.DEVICE, False, uniform)
        field, _ = _render(self.DEVICE, True, uniform)
        self.assertTrue(torch.allclose(field, pooled, atol=1e-6))

    def test_distinct_corners_vary_across_the_face(self):
        """The min flattens the face to its most transparent corner; the field
        is never below it and differs wherever the corners differ."""
        pooled, _ = _render(self.DEVICE, False, OPACITIES)
        field, _ = _render(self.DEVICE, True, OPACITIES)
        coverage = pooled.sum(0) > 0
        self.assertGreater(int(coverage.sum()), 100)
        self.assertFalse(torch.allclose(field, pooled, atol=1e-4))
        self.assertTrue(bool((field.sum(0)[coverage] >= pooled.sum(0)[coverage] - 1e-6).all()))

    def test_every_corner_receives_gradient(self):
        image, weights = _render(self.DEVICE, True, OPACITIES, grad=True)
        _loss(image).backward()
        for i in range(3):
            self.assertNotEqual(float(weights.grad[i]), 0.0, f"corner {i} starved")

    def test_gradient_matches_finite_differences(self):
        """The contribution is a model change with its exact derivative, not a
        surrogate: the analytic gradient must equal the numerical one."""
        image, weights = _render(self.DEVICE, True, OPACITIES, grad=True)
        _loss(image).backward()
        step = 0.01
        for i in range(3):
            up = list(OPACITIES)
            down = list(OPACITIES)
            up[i] += step
            down[i] -= step
            numeric = (_loss(_render(self.DEVICE, True, up)[0])
                       - _loss(_render(self.DEVICE, True, down)[0])) / (2 * step)
            analytic = float(weights.grad[i])
            self.assertAlmostEqual(
                analytic, float(numeric), delta=2e-3 * max(1.0, abs(analytic)),
                msg=f"corner {i}: analytic {analytic} vs numeric {float(numeric)}",
            )

    def test_min_path_is_untouched(self):
        """Off, the gradient still reaches the argmin alone, as published."""
        image, weights = _render(self.DEVICE, False, OPACITIES, grad=True)
        _loss(image).backward()
        self.assertNotEqual(float(weights.grad[0]), 0.0)
        self.assertEqual(float(weights.grad[1]), 0.0)
        self.assertEqual(float(weights.grad[2]), 0.0)

    def test_field_is_deterministic_in_the_forward(self):
        first, _ = _render(self.DEVICE, True, OPACITIES)
        second, _ = _render(self.DEVICE, True, OPACITIES)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
