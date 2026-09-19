"""Opacity-aware topology survival (OATS): native rasterizer contract.

The kernel change is one ``atomicAdd(integrated_blending + j_id, blending_weight)``
beside the existing ``atomicMax``, into a caller-owned buffer passed through
the raster settings. These tests prove the accumulator is the integral of
``alpha * T`` and that adding it perturbs nothing v1 depends on.

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

ACCUMULATOR_SUPPORTED = RASTERIZER_AVAILABLE and "integrated_blending" in getattr(
    TriangleRasterizationSettings, "_fields", ()
)

SIZE = 64
# Two overlapping faces at different depths, so the back face is partly
# occluded and its T is below one where they overlap.
VERTICES = (
    (-0.62, -0.52, 2.0), (0.62, -0.52, 2.0), (0.0, 0.64, 2.0),
    (-0.40, -0.70, 3.0), (0.80, -0.30, 3.0), (0.20, 0.80, 3.0),
)
FACES = ((0, 1, 2), (3, 4, 5))
OPACITIES = (0.6, 0.6, 0.6, 0.7, 0.7, 0.7)
GREY = 0.5


def _render(device, buffer=None, faces=FACES, colors=None):
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
        integrated_blending=buffer,
    )
    if colors is None:
        colors = torch.full((len(VERTICES), 3), GREY, dtype=torch.float32, device=device)
    return TriangleRasterizer(settings)(
        vertices=torch.tensor(VERTICES, dtype=torch.float32, device=device),
        triangles_indices=torch.tensor(faces, dtype=torch.int32, device=device),
        vertex_weights=torch.tensor(OPACITIES, dtype=torch.float32, device=device),
        sigma=1.0,
        scaling=torch.zeros(1, dtype=torch.float32, device=device),
        colors_precomp=colors,
    )


def _zeros(faces=len(FACES)):
    return torch.zeros(faces, dtype=torch.float32, device="cuda")


@unittest.skipUnless(
    ACCUMULATOR_SUPPORTED and torch.cuda.is_available(),
    "needs a CUDA device and a rasterizer rebuilt with integrated_blending",
)
class IntegratedBlendingTest(unittest.TestCase):
    DEVICE = "cuda"

    def test_every_output_is_unchanged_by_the_accumulator(self):
        """Colors, depth, radii, max_blending and was_rendered with the buffer
        on must equal the buffer-off render bitwise: v1 depends on all five."""
        off = _render(self.DEVICE)
        on = _render(self.DEVICE, buffer=_zeros())
        for index, (a, b) in enumerate(zip(off, on)):
            if index == 3:
                # With precomputed colours the rasterizer never fills the vertex
                # depths (rasterizer_impl.cu computes them with the SH colours
                # only), so the depth and median-depth channels read unset
                # memory here. Training always takes the SH path.
                a, b = a[[1, 2, 3, 4, 6]], b[[1, 2, 3, 4, 6]]
            self.assertTrue(torch.equal(a, b), f"output {index} changed")

    def test_integral_equals_the_summed_weight_of_a_lone_face(self):
        """Alone on a black background, a uniformly grey face renders
        alpha * grey in every pixel with T = 1, so its integral is the image
        sum divided by grey."""
        buffer = _zeros(1)
        image = _render(self.DEVICE, buffer=buffer, faces=FACES[:1])[0]
        expected = float(image[0].sum()) / GREY
        self.assertGreater(expected, 10.0)
        self.assertAlmostEqual(float(buffer[0]), expected, delta=1e-3 * expected)

    def test_integral_dominates_the_max(self):
        """The sum over pixels contains the max term."""
        buffer = _zeros()
        max_blending = _render(self.DEVICE, buffer=buffer)[4]
        self.assertTrue(bool((buffer >= max_blending - 1e-6).all()))
        self.assertTrue(bool((buffer > 0).all()))

    def test_occlusion_lowers_the_back_face(self):
        """The back face alone versus behind the front face: T < 1 where they
        overlap, so its integral must drop."""
        alone = _zeros(1)
        _render(self.DEVICE, buffer=alone, faces=FACES[1:])
        both = _zeros()
        _render(self.DEVICE, buffer=both)
        self.assertLess(float(both[1]), float(alone[0]))

    def test_reusing_the_buffer_sums_over_views(self):
        """The cleanup pass hands one buffer to every view."""
        once = _zeros()
        _render(self.DEVICE, buffer=once)
        twice = _zeros()
        _render(self.DEVICE, buffer=twice)
        _render(self.DEVICE, buffer=twice)
        self.assertTrue(torch.allclose(twice, 2 * once, rtol=1e-5))

    def test_a_buffer_of_the_wrong_size_is_refused(self):
        with self.assertRaises(RuntimeError):
            _render(self.DEVICE, buffer=_zeros(3))


if __name__ == "__main__":
    unittest.main()
