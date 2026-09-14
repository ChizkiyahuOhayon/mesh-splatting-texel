"""Opacity-aware topology survival (OATS): native rasterizer contract.

The kernel change is one `atomicAdd(integrated_blending + j_id, blending_weight)`
beside the existing `atomicMax` at forward.cu:767, behind a null-pointer guard.
These tests prove (a) the new accumulator is correct, and (b) adding it did not
perturb anything the v1 results depend on.

They need the rebuilt native extension and a CUDA device, so they skip on the
Mac and must be run on the A40 before any training with --survival_cleanup.
See handover/OATS_PLAN.md §5.
"""

import unittest

import torch

try:
    from diff_triangle_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    RASTERIZER_AVAILABLE = True
except Exception:  # noqa: BLE001 - native extension absent on the laptop
    RASTERIZER_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()

INTEGRATED_BLENDING_IMPLEMENTED = RASTERIZER_AVAILABLE and hasattr(
    GaussianRasterizationSettings, "_fields"
) and "accumulate_integrated_blending" in getattr(
    GaussianRasterizationSettings, "_fields", ()
)

_SKIP = (
    "needs a CUDA device and a rebuilt rasterizer exporting "
    "integrated_blending (handover/OATS_PLAN.md §4)"
)


@unittest.skipUnless(
    RASTERIZER_AVAILABLE and CUDA_AVAILABLE and INTEGRATED_BLENDING_IMPLEMENTED,
    _SKIP,
)
class IntegratedBlendingTest(unittest.TestCase):
    """`S_f` must equal the analytic sum of alpha*T, and must not disturb the
    statistics v1 already depends on."""

    def test_buffer_shape_and_dtype_when_enabled(self):
        """integrated_blending is [F] float32 on the render device."""
        raise NotImplementedError(
            "fill in from tests/test_rasterizer*.py once the kernel is built")

    def test_absent_or_unwritten_when_disabled(self):
        """With the flag off the kernel must take the null-pointer branch: no
        allocation, or an all-zero buffer that the kernel never touched."""
        raise NotImplementedError(
            "fill in from tests/test_rasterizer*.py once the kernel is built")

    def test_matches_analytic_alpha_times_T_on_a_two_face_scene(self):
        """Hand-built scene with known alpha and T: S_f equals the summed
        alpha*T to float tolerance, for both the front and the occluded face."""
        raise NotImplementedError(
            "fill in from tests/test_rasterizer*.py once the kernel is built")

    def test_integral_dominates_the_max(self):
        """S_f >= max_blending_f always, since the sum contains the max term.
        This is the cheapest end-to-end sanity check on a real scene."""
        raise NotImplementedError(
            "fill in from tests/test_rasterizer*.py once the kernel is built")

    def test_max_blending_is_unchanged_by_the_new_accumulator(self):
        """The whole v1 cleanup depends on max_blending. With the flag ON, it
        must still equal the value the flag-off path produces, bitwise."""
        raise NotImplementedError(
            "fill in from tests/test_rasterizer*.py once the kernel is built")

    def test_flag_off_is_bitwise_identical_to_the_v1_build(self):
        """Colors, depth, radii, max_blending and was_rendered on a fixed scene
        and camera, against values recorded from the pre-change build. This is
        the guard that lets the frozen v1 numbers stay comparable."""
        raise NotImplementedError(
            "record reference tensors from 62233b6 and compare")


if __name__ == "__main__":
    unittest.main()
