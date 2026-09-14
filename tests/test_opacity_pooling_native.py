"""Softmin opacity routing: native rasterizer contract.

The kernel change lives in the backward pass only (``backward.cu``): a face's
opacity gradient is split over its three vertices instead of going entirely to
the argmin. These tests pin the three properties the method rests on, on a mesh
whose three vertex opacities are deliberately distinct so the routing is
observable:

1. the forward is untouched at every temperature;
2. the split conserves the total gradient mass, so nothing is amplified;
3. ``beta = 0`` is the published argmin routing, and a large beta converges back
   to it, which makes the published method a strict special case.

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

BETA_SUPPORTED = RASTERIZER_AVAILABLE and "opacity_pool_beta" in getattr(
    TriangleRasterizationSettings, "_fields", ()
)

SIZE = 64
# The three opacities of the single face, chosen distinct and inside the trained
# range so every softmin weight is resolvable in float32.
OPACITIES = (0.81, 0.88, 0.97)


def _settings(device, beta):
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
        opacity_pool_beta=beta,
    )


def _render_and_backward(device, beta, seed=0):
    """Render one face and return (image, per-vertex opacity gradient)."""
    vertices = torch.tensor(
        [[-0.62, -0.52, 2.0], [0.62, -0.52, 2.0], [0.0, 0.64, 2.0]],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    weights = torch.tensor(
        OPACITIES, dtype=torch.float32, device=device
    ).requires_grad_(True)
    colors = torch.tensor(
        [[0.20, 0.34, 0.42], [0.43, 0.18, 0.12], [0.11, 0.51, 0.25]],
        dtype=torch.float32,
        device=device,
    )
    rasterizer = TriangleRasterizer(_settings(device, beta))
    image = rasterizer(
        vertices=vertices,
        triangles_indices=faces,
        vertex_weights=weights,
        sigma=1.0,
        scaling=torch.zeros(1, dtype=torch.float32, device=device),
        colors_precomp=colors,
    )[0]
    # A fixed, asymmetric target keeps the upstream gradient deterministic and
    # non-zero without depending on the pixel layout.
    torch.manual_seed(seed)
    target = torch.rand_like(image)
    (image - target).abs().mean().backward()
    return image.detach(), weights.grad.detach().clone()


@unittest.skipUnless(
    BETA_SUPPORTED and torch.cuda.is_available(),
    "needs a CUDA device and a rasterizer rebuilt with opacity_pool_beta",
)
class SoftminRoutingNativeTest(unittest.TestCase):
    DEVICE = "cuda"

    def test_forward_is_identical_at_every_temperature(self):
        """The contribution is a gradient change. If the image moves, the
        published forward model has been altered and every frozen number in the
        paper stops being comparable."""
        reference, _ = _render_and_backward(self.DEVICE, 0.0)
        for beta in (1.0, 5.0, 50.0, 1000.0):
            image, _ = _render_and_backward(self.DEVICE, beta)
            self.assertTrue(
                torch.equal(image, reference),
                f"forward changed at beta={beta}",
            )

    def test_hard_routing_reaches_only_the_argmin(self):
        _, grad = _render_and_backward(self.DEVICE, 0.0)
        self.assertNotEqual(float(grad[0]), 0.0)
        self.assertEqual(float(grad[1]), 0.0)
        self.assertEqual(float(grad[2]), 0.0)

    def test_softmin_reaches_every_vertex(self):
        """The defect being fixed: two of three vertices learn nothing."""
        _, grad = _render_and_backward(self.DEVICE, 5.0)
        for i in range(3):
            self.assertNotEqual(float(grad[i]), 0.0, f"vertex {i} still starved")

    def test_total_gradient_mass_is_conserved(self):
        """Sum over vertices must match the hard-routing sum: the split is a
        redistribution, never a gain term."""
        _, hard = _render_and_backward(self.DEVICE, 0.0)
        for beta in (1.0, 5.0, 50.0):
            _, soft = _render_and_backward(self.DEVICE, beta)
            self.assertAlmostEqual(
                float(soft.sum()), float(hard.sum()), places=4,
                msg=f"gradient mass not conserved at beta={beta}",
            )

    def test_large_beta_converges_to_hard_routing(self):
        _, hard = _render_and_backward(self.DEVICE, 0.0)
        _, soft = _render_and_backward(self.DEVICE, 1e5)
        self.assertTrue(torch.allclose(soft, hard, atol=1e-6))

    def test_weights_follow_the_declared_softmin(self):
        """The realized split must equal exp(-beta*(o - o_min)) normalized, so
        the CPU schedule in sota/opacity_pooling.py describes the kernel."""
        from sota.opacity_pooling import softmin_weights

        beta = 8.0
        _, grad = _render_and_backward(self.DEVICE, beta)
        expected = softmin_weights(list(OPACITIES), beta)
        total = float(grad.sum())
        for i in range(3):
            self.assertAlmostEqual(
                float(grad[i]) / total, expected[i], places=4,
                msg=f"vertex {i} weight disagrees with the schedule",
            )

    def test_ordering_is_preserved(self):
        """The most transparent vertex keeps the largest share at every beta."""
        for beta in (1.0, 10.0, 100.0):
            _, grad = _render_and_backward(self.DEVICE, beta)
            share = [abs(float(g)) for g in grad]
            self.assertEqual(max(range(3), key=lambda i: share[i]), 0)


if __name__ == "__main__":
    unittest.main()
