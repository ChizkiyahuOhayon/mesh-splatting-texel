import unittest

import torch

from rits_prolongation import (
    DONOR_OPACITY,
    DONOR_WINDOW,
    donor_mode,
    install_prolongation_probe,
)


class DummyModel:
    def __init__(self):
        self.vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
            requires_grad=True,
        )
        self.vertex_weight = torch.zeros(4, 1, requires_grad=True)
        self._features_dc = torch.arange(12.0).reshape(4, 1, 3).requires_grad_(True)
        self._features_rest = torch.arange(24.0).reshape(4, 2, 3).requires_grad_(True)
        self._triangle_indices = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32)
        self.texel_order = 0
        self.opacity_floor = 0.0
        self.eps = 1e-6
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit
        self.optimizer = object()
        self.image_size = torch.zeros(2)
        self.importance_score = torch.zeros(2)
        self.pixel_count = torch.zeros(2, dtype=torch.int)

    def validate_face_state(self):
        assert self.image_size.shape[0] == self._triangle_indices.shape[0]


class DonorModeTest(unittest.TestCase):
    def test_bitmask_matches_kernel_contract(self):
        self.assertEqual(donor_mode(False, False), 0)
        self.assertEqual(donor_mode(True, False), DONOR_WINDOW)
        self.assertEqual(donor_mode(False, True), DONOR_OPACITY)
        self.assertEqual(donor_mode(True, True), DONOR_WINDOW | DONOR_OPACITY)


class InstallProlongationProbeTest(unittest.TestCase):
    def test_children_point_at_their_parent_corners(self):
        model = DummyModel()
        probe = install_prolongation_probe(
            model, torch.tensor([0]), donor_window=True, donor_opacity=True
        )

        window_source, donor_indices, mode = probe["window_donors"]
        self.assertEqual(mode, DONOR_WINDOW | DONOR_OPACITY)
        self.assertEqual(probe["child_face_start"], 1)
        self.assertEqual(window_source.dtype, torch.int32)
        self.assertEqual(donor_indices.dtype, torch.int32)
        self.assertEqual(window_source.shape[0], model._triangle_indices.shape[0])
        self.assertTrue(torch.equal(window_source[:1], torch.tensor([-1], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(window_source[1:], torch.zeros(4, dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(donor_indices, torch.tensor([[0, 1, 2]], dtype=torch.int32))
        )

    def test_disabled_donors_still_install_the_split(self):
        model = DummyModel()
        probe = install_prolongation_probe(
            model, torch.tensor([0]), donor_window=False, donor_opacity=False
        )
        self.assertIsNone(probe["window_donors"])
        self.assertTrue(probe["topology_valid"])
        self.assertTrue(probe["prefix_unchanged"])


def project(point, matrix, width, height):
    hom = matrix @ torch.cat([point, torch.ones(1)])
    inv_w = 1.0 / (hom[3] + 1e-7)
    ndc = hom[:2] * inv_w
    return torch.stack(
        [((ndc[0] + 1.0) * width - 1.0) * 0.5, ((ndc[1] + 1.0) * height - 1.0) * 0.5]
    )


def window_lines(tri2d):
    """Edge lines and incenter distance exactly as the CUDA preprocess builds them."""
    a = (tri2d[1] - tri2d[2]).norm()
    b = (tri2d[0] - tri2d[2]).norm()
    c = (tri2d[0] - tri2d[1]).norm()
    incenter = (a * tri2d[0] + b * tri2d[1] + c * tri2d[2]) / (a + b + c)
    normals, offsets, dist = [], [], None
    for i in range(3):
        p1, p2 = tri2d[i], tri2d[(i + 1) % 3]
        normal = torch.stack([p2[1] - p1[1], -(p2[0] - p1[0])])
        normal = normal / normal.norm()
        offset = -(normal @ p1)
        dist = normal @ incenter + offset
        if dist > 0:
            normal, offset, dist = -normal, -offset, -dist
        normals.append(normal)
        offsets.append(offset)
    return torch.stack(normals), torch.stack(offsets), dist


def splat_alpha(pixel, tri2d, min_weight, sigma, window_tri2d=None):
    """Per-pixel alpha exactly as renderCUDA composes it, with optional donor."""
    own_normals, own_offsets, own_dist = window_lines(tri2d)
    own_dists = own_normals @ pixel + own_offsets
    if bool((own_dists > 0).any()):
        return torch.tensor(0.0)
    if window_tri2d is None:
        max_val, inv_dist = own_dists.max(), 1.0 / own_dist
    else:
        donor_normals, donor_offsets, donor_dist = window_lines(window_tri2d)
        max_val = (donor_normals @ pixel + donor_offsets).max().clamp(max=0.0)
        inv_dist = 1.0 / donor_dist
    phi = max_val * inv_dist
    return min_weight * phi.clamp(min=0.0) ** sigma


class ProlongationIdentityTest(unittest.TestCase):
    """The window math contract: at exact midpoint initialization, a child
    evaluated with its parent's window and opacity reproduces the parent's
    per-pixel alpha, while the four children partition the parent's support."""

    def setUp(self):
        torch.manual_seed(0)
        # Perspective with visible depth variation so the projected midpoints
        # are off the screen midpoints and the identity is not trivially affine.
        self.matrix = torch.tensor(
            [
                [1.2, 0.1, 0.3, 0.0],
                [0.0, 1.1, -0.2, 0.1],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.9, 0.5],
            ]
        )
        self.sigma = 2.0
        parent3d = torch.tensor(
            [[-0.4, -0.3, 1.6], [0.5, -0.2, 2.4], [0.0, 0.6, 2.0]]
        )
        weights = torch.tensor([0.9, 0.6, 0.75])
        midpoints3d = 0.5 * (parent3d + parent3d.roll(-1, dims=0))
        m01, m12, m20 = midpoints3d
        self.children3d = [
            torch.stack([parent3d[0], m01, m20]),
            torch.stack([parent3d[1], m12, m01]),
            torch.stack([parent3d[2], m20, m12]),
            torch.stack([m01, m12, m20]),
        ]
        self.parent2d = self._project_triangle(parent3d)
        self.children2d = [self._project_triangle(child) for child in self.children3d]
        self.parent_weight = weights.min()

    def _project_triangle(self, tri3d):
        return torch.stack([project(p, self.matrix, 64, 64) for p in tri3d])

    def _interior_pixels(self):
        normals, offsets, _ = window_lines(self.parent2d)
        low = self.parent2d.min(dim=0).values.floor()
        high = self.parent2d.max(dim=0).values.ceil()
        xs = torch.arange(low[0], high[0] + 1)
        ys = torch.arange(low[1], high[1] + 1)
        pixels = torch.cartesian_prod(xs, ys).float()
        margin = 1e-3
        keep = []
        for pixel in pixels:
            inside = bool(((normals @ pixel + offsets) < -margin).all())
            near_child_edge = any(
                bool((window_lines(child)[0] @ pixel + window_lines(child)[1]).abs().min() < margin)
                for child in self.children2d
            )
            if inside and not near_child_edge:
                keep.append(pixel)
        return keep

    def test_children_partition_and_reproduce_parent_alpha(self):
        pixels = self._interior_pixels()
        self.assertGreater(len(pixels), 50)
        variant1_differs = False
        for pixel in pixels:
            owners = [
                index
                for index, child in enumerate(self.children2d)
                if splat_alpha(pixel, child, self.parent_weight, self.sigma) >= 0
                and bool(((window_lines(child)[0] @ pixel + window_lines(child)[1]) <= 0).all())
            ]
            self.assertEqual(len(owners), 1, f"pixel {pixel} owned by {owners}")
            child = self.children2d[owners[0]]
            parent_alpha = splat_alpha(
                pixel, self.parent2d, self.parent_weight, self.sigma
            )
            child_alpha = splat_alpha(
                pixel, child, self.parent_weight, self.sigma, window_tri2d=self.parent2d
            )
            self.assertLess(float((child_alpha - parent_alpha).abs()), 1e-5)
            own_alpha = splat_alpha(pixel, child, self.parent_weight, self.sigma)
            if float((own_alpha - parent_alpha).abs()) > 1e-4:
                variant1_differs = True
        self.assertTrue(
            variant1_differs,
            "child-local windows should not reproduce the parent alpha",
        )


if __name__ == "__main__":
    unittest.main()
