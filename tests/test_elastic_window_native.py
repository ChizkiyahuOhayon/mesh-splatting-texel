"""Elastic window: native rasterizer contract.

The published window phi^sigma is zero on every edge and outside the face. The
elastic window exp(-sigma x^(2/sigma)) of Elastic Triangle Splatting has support
on both sides of each edge and an edge value exp(-sigma) that tends to one. These
tests pin what the method rests on:

1. at the sharp end of the schedule both windows draw the same opaque face;
2. early in the schedule the elastic window reaches beyond the edges;
3. its position gradient points the way finite differences do, and stays finite
   in the sharp limit where the powers overflow;
4. a pixel outside the face now moves the vertices, which the published window
   cannot do.

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

ELASTIC_SUPPORTED = RASTERIZER_AVAILABLE and "elastic_window" in getattr(
    TriangleRasterizationSettings, "_fields", ()
)

SIZE = 64
VERTICES = ((-0.50, -0.42, 2.0), (0.50, -0.42, 2.0), (0.0, 0.52, 2.0))
COLORS = ((0.20, 0.34, 0.42), (0.43, 0.18, 0.12), (0.11, 0.51, 0.25))


def _render(device, elastic, sigma, vertices=VERTICES, colors=COLORS):
    field_of_view = 1.0
    settings = TriangleRasterizationSettings(
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
        elastic_window=elastic,
    )
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    return TriangleRasterizer(settings)(
        vertices=vertices,
        triangles_indices=torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device),
        vertex_weights=torch.full((3,), 0.9, dtype=torch.float32, device=device),
        sigma=sigma,
        scaling=torch.zeros(1, dtype=torch.float32, device=device),
        colors_precomp=torch.tensor(colors, dtype=torch.float32, device=device),
    )[0]


def _loss(image):
    probe = torch.linspace(-1.0, 1.0, image.numel(), device=image.device)
    return (image.flatten() * probe).sum()


@unittest.skipUnless(
    ELASTIC_SUPPORTED and torch.cuda.is_available(),
    "needs a CUDA device and a rasterizer rebuilt with elastic_window",
)
class ElasticWindowNativeTest(unittest.TestCase):
    DEVICE = "cuda"

    def test_sharp_limit_matches_the_published_window(self):
        """Both windows anneal to the opaque step, so at sigma = 1e-4 they draw
        the same face; only a one-pixel rim along the edges may differ."""
        published = _render(self.DEVICE, False, 1e-4)
        elastic = _render(self.DEVICE, True, 1e-4)
        coverage = published.sum(0) > 0
        differ = (published - elastic).abs().amax(0) > 1e-3
        self.assertGreater(int(coverage.sum()), 200)
        self.assertLess(float(differ.float().sum()) / float(coverage.sum()), 0.10)

    def test_soft_window_reaches_beyond_the_edges(self):
        """At sigma = 1 the published window deposits nothing outside the face;
        the elastic one does, which is its whole point."""
        published = _render(self.DEVICE, False, 1.0).sum(0) > 0
        elastic = _render(self.DEVICE, True, 1.0).sum(0) > 0
        self.assertTrue(bool((elastic | ~published).all()), "elastic lost interior pixels")
        self.assertGreater(int((elastic & ~published).sum()), 50)

    def test_edge_value_tends_to_one(self):
        """The coverage of the interior rises as sigma anneals, instead of the
        published window's persistent fade to zero at the edge."""
        soft = _render(self.DEVICE, True, 1.0).sum()
        sharp = _render(self.DEVICE, True, 1e-2).sum()
        self.assertGreater(float(sharp), float(soft))

    def _window_gradient_cosine(self, elastic):
        """Cosine between the analytic and numerical in-plane vertex gradient.

        A uniform colour removes the colour-interpolation term, which the
        published backward drops with screen_space_gradients off, so only the
        window moves the image. Both windows also omit the inradius term of the
        derivative, so the comparison is of direction, not size."""
        uniform = ((0.3, 0.3, 0.3),) * 3
        base = torch.tensor(VERTICES, dtype=torch.float32, device=self.DEVICE)
        vertices = base.clone().requires_grad_(True)
        _loss(_render(self.DEVICE, elastic, 1.0, vertices, uniform)).backward()
        analytic = vertices.grad[:, :2].flatten()
        numeric = torch.zeros_like(analytic)
        step = 2e-3
        for index in range(analytic.numel()):
            row, col = divmod(index, 2)
            up = base.clone()
            down = base.clone()
            up[row, col] += step
            down[row, col] -= step
            numeric[index] = (_loss(_render(self.DEVICE, elastic, 1.0, up, uniform))
                              - _loss(_render(self.DEVICE, elastic, 1.0, down, uniform))) / (2 * step)
        return float(torch.nn.functional.cosine_similarity(analytic, numeric, dim=0)), analytic, numeric

    def test_position_gradient_follows_finite_differences(self):
        cosine, analytic, numeric = self._window_gradient_cosine(True)
        published, _, _ = self._window_gradient_cosine(False)
        print(f"window-gradient cosine: elastic {cosine:.3f}, published {published:.3f}")
        self.assertGreater(cosine, 0.8, f"analytic {analytic} numeric {numeric}")

    def test_gradients_stay_finite_in_the_sharp_limit(self):
        vertices = torch.tensor(VERTICES, dtype=torch.float32, device=self.DEVICE)
        vertices.requires_grad_(True)
        _loss(_render(self.DEVICE, True, 1e-4, vertices)).backward()
        self.assertTrue(bool(torch.isfinite(vertices.grad).all()))

    def test_exterior_pixels_move_the_vertices(self):
        """A loss living only outside the face: the published window gives the
        vertices nothing, the elastic window a non-zero push."""
        outside = _render(self.DEVICE, False, 1.0).sum(0) == 0

        def exterior_loss(elastic):
            vertices = torch.tensor(VERTICES, dtype=torch.float32, device=self.DEVICE)
            vertices.requires_grad_(True)
            image = _render(self.DEVICE, elastic, 1.0, vertices)
            (image.sum(0) * outside).sum().backward()
            return float(vertices.grad.abs().sum())

        self.assertEqual(exterior_loss(False), 0.0)
        self.assertGreater(exterior_loss(True), 0.0)


if __name__ == "__main__":
    unittest.main()
