import unittest

import torch

from edgeval_core import (
    build_edge_topology,
    crossfit_edge_value,
    deterministic_camera_folds,
    edge_basis,
    exact_squared_loss_gain,
    p2_sh1_edge_radiance,
    vertex_sh1_factors,
)


class EdgeTopologyTest(unittest.TestCase):
    def test_local_rows_match_the_declared_edge_order(self):
        topology = build_edge_topology(torch.tensor([[4, 1, 3]], dtype=torch.int32))
        self.assertEqual(topology.edge_vertices.tolist(), [[1, 3], [1, 4], [3, 4]])
        self.assertEqual(topology.face_edges.tolist(), [[1, 0, 2]])
        self.assertEqual(topology.edge_face_count.tolist(), [1, 1, 1])

    def test_two_faces_share_one_global_edge_despite_reversed_orientation(self):
        topology = build_edge_topology(torch.tensor([[0, 1, 2], [2, 1, 3]]))
        shared_first = int(topology.face_edges[0, 1])
        shared_second = int(topology.face_edges[1, 0])
        self.assertEqual(shared_first, shared_second)
        self.assertEqual(topology.edge_vertices[shared_first].tolist(), [1, 2])
        self.assertEqual(int(topology.edge_face_count[shared_first]), 2)

    def test_boundary_edges_are_allowed(self):
        topology = build_edge_topology(torch.tensor([[0, 1, 2]]))
        self.assertEqual(topology.edge_face_count.tolist(), [1, 1, 1])

    def test_degenerate_faces_are_refused(self):
        with self.assertRaisesRegex(ValueError, "repeats a vertex"):
            build_edge_topology(torch.tensor([[0, 1, 1]]))

    def test_nonmanifold_edges_are_refused(self):
        faces = torch.tensor([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
        with self.assertRaisesRegex(ValueError, "non-manifold edge"):
            build_edge_topology(faces)

    def test_empty_mesh_has_well_typed_empty_tables(self):
        topology = build_edge_topology(torch.empty((0, 3), dtype=torch.int32))
        self.assertEqual(tuple(topology.edge_vertices.shape), (0, 2))
        self.assertEqual(tuple(topology.face_edges.shape), (0, 3))
        self.assertEqual(topology.edge_vertices.dtype, torch.int64)


class EdgeBasisTest(unittest.TestCase):
    def test_midpoint_of_one_edge_activates_only_its_basis(self):
        value = edge_basis(torch.tensor([0.5, 0.5, 0.0]))
        self.assertTrue(torch.equal(value, torch.tensor([1.0, 0.0, 0.0])))

    def test_vertices_activate_no_hierarchical_edge_detail(self):
        self.assertTrue(torch.equal(edge_basis(torch.eye(3)), torch.zeros((3, 3))))

    def test_shared_edge_value_is_orientation_invariant(self):
        forward = edge_basis(torch.tensor([0.25, 0.75, 0.0]))[0]
        reversed_ = edge_basis(torch.tensor([0.75, 0.25, 0.0]))[0]
        self.assertEqual(float(forward), float(reversed_))


class AngularEdgeBasisTest(unittest.TestCase):
    def test_stock_degree_one_factor_convention(self):
        factors = vertex_sh1_factors(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.zeros(3),
        )
        c1 = 0.4886025119029199
        expected = torch.tensor([[0.0, 0.0, -c1], [-c1, 0.0, 0.0], [0.0, c1, 0.0]])
        self.assertTrue(torch.allclose(factors, expected))

    def test_vertices_activate_no_angular_edge_detail(self):
        factors = torch.randn(3, 3, generator=torch.Generator().manual_seed(3))
        coefficients = torch.randn(4, 3, generator=torch.Generator().manual_seed(5))
        for vertex in torch.eye(3):
            value = p2_sh1_edge_radiance(vertex, factors, coefficients, 0)
            self.assertTrue(torch.equal(value, torch.zeros(3)))

    def test_shared_edge_is_continuous_under_reversed_orientation(self):
        endpoints = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
        camera = torch.tensor([0.2, -0.4, 0.1])
        factors_left = vertex_sh1_factors(
            torch.cat((endpoints, torch.tensor([[0.0, 1.0, 2.0]]))), camera
        )
        factors_right = vertex_sh1_factors(
            torch.cat((endpoints.flip(0), torch.tensor([[0.0, -1.0, 2.0]]))), camera
        )
        coefficients = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 11.0
        left = p2_sh1_edge_radiance(
            torch.tensor([0.25, 0.75, 0.0]), factors_left, coefficients, 0
        )
        right = p2_sh1_edge_radiance(
            torch.tensor([0.75, 0.25, 0.0]), factors_right, coefficients, 0
        )
        self.assertTrue(torch.allclose(left, right, atol=0.0, rtol=0.0))

    def test_camera_motion_changes_only_angular_rows(self):
        vertices = torch.tensor([[-0.5, 0.0, 2.0], [0.5, 0.0, 2.0], [0.0, 0.7, 2.0]])
        coefficients = torch.zeros(4, 3)
        coefficients[1:, :] = torch.eye(3)
        barycentric = torch.tensor([0.5, 0.5, 0.0])
        first = p2_sh1_edge_radiance(
            barycentric, vertex_sh1_factors(vertices, torch.zeros(3)), coefficients, 0
        )
        second = p2_sh1_edge_radiance(
            barycentric,
            vertex_sh1_factors(vertices, torch.tensor([0.7, -0.2, 0.1])),
            coefficients,
            0,
        )
        self.assertFalse(torch.equal(first, second))


class CameraFoldTest(unittest.TestCase):
    def test_mapping_is_independent_of_input_order(self):
        names = ["DSC3", "DSC1", "DSC4", "DSC2", "DSC5"]
        first = deterministic_camera_folds(names, 4)
        second = deterministic_camera_folds(list(reversed(names)), 4)
        self.assertEqual(first, second)
        self.assertEqual(first, {"DSC1": 0, "DSC2": 1, "DSC3": 2, "DSC4": 3, "DSC5": 0})

    def test_duplicate_camera_identity_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            deterministic_camera_folds(["same", "same"], 4)


class ExactGainTest(unittest.TestCase):
    def test_identity_matches_direct_squared_loss_difference(self):
        generator = torch.Generator().manual_seed(17)
        residual = torch.randn(31, 3, generator=generator, dtype=torch.float64)
        correction = torch.randn(31, 3, generator=generator, dtype=torch.float64)
        direct = residual.square().sum() - (residual - correction).square().sum()
        self.assertTrue(torch.allclose(exact_squared_loss_gain(residual, correction), direct))

    def test_harmful_correction_stays_negative(self):
        residual = torch.zeros(2, 3)
        correction = torch.ones(2, 3)
        self.assertEqual(float(exact_squared_loss_gain(residual, correction)), -6.0)


def _isotropic_fold_stats(rhs_rows):
    rhs = torch.tensor(rhs_rows, dtype=torch.float64)
    fold_count = rhs.shape[-2]
    gram = torch.eye(3, dtype=torch.float64).expand(*rhs.shape[:-1], 3, 3).clone()
    rss = torch.full(rhs.shape[:-1], 10.0, dtype=torch.float64)
    observations = torch.full(rhs.shape[:-1], 30, dtype=torch.int64)
    assert gram.shape[-3] == fold_count
    return gram, rhs, rss, observations


class CrossFitValueTest(unittest.TestCase):
    def test_consistent_edge_has_positive_value(self):
        stats = _isotropic_fold_stats([[1.0, 0.0, 0.0]] * 4)
        result = crossfit_edge_value(*stats)
        self.assertTrue(bool(result.valid))
        self.assertGreater(float(result.value), 0.0)
        self.assertTrue(torch.all(result.fold_gain > 0))

    def test_signed_fold_failures_are_not_clamped(self):
        stats = _isotropic_fold_stats(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-8.0, 0.0, 0.0]]
        )
        result = crossfit_edge_value(*stats)
        self.assertTrue(bool(result.valid))
        self.assertTrue(bool((result.fold_gain < 0).any()))
        self.assertLess(float(result.value), float(result.mean_gain))

    def test_zero_design_is_invalid_not_zero_score(self):
        gram = torch.zeros(4, 3, 3, dtype=torch.float64)
        rhs = torch.zeros(4, 3, dtype=torch.float64)
        rss = torch.ones(4, dtype=torch.float64)
        observations = torch.full((4,), 30)
        result = crossfit_edge_value(gram, rhs, rss, observations)
        self.assertFalse(bool(result.valid))
        self.assertTrue(bool(torch.isnan(result.value)))

    def test_batch_rows_equal_independent_calls(self):
        rhs = [
            [[1.0, 0.0, 0.0]] * 4,
            [[0.5, 0.25, 0.0], [0.6, 0.2, 0.0], [0.4, 0.3, 0.0], [0.5, 0.25, 0.0]],
        ]
        stats = _isotropic_fold_stats(rhs)
        batch = crossfit_edge_value(*stats)
        for row in range(2):
            single = crossfit_edge_value(*(value[row] for value in stats))
            self.assertTrue(torch.equal(batch.valid[row], single.valid))
            self.assertTrue(torch.allclose(batch.value[row], single.value))
            self.assertTrue(torch.allclose(batch.fold_gain[row], single.fold_gain))

    def test_gcv_exact_tie_prefers_larger_ridge(self):
        gram = torch.eye(3, dtype=torch.float64).expand(4, 3, 3).clone()
        rhs = torch.zeros(4, 3, dtype=torch.float64)
        rss = torch.zeros(4, dtype=torch.float64)
        observations = torch.full((4,), 30)
        result = crossfit_edge_value(gram, rhs, rss, observations)
        expected = (3.0 * torch.eye(3).trace() / 3.0) * 1e2
        self.assertTrue(torch.allclose(result.ridge, torch.full_like(result.ridge, expected)))

    def test_invalid_shapes_are_refused(self):
        with self.assertRaisesRegex(ValueError, "fold_gram"):
            crossfit_edge_value(
                torch.zeros(4, 2, 2), torch.zeros(4, 3), torch.zeros(4), torch.ones(4)
            )

    def test_mismatched_rhs_axes_are_refused(self):
        with self.assertRaisesRegex(ValueError, "fold_rhs"):
            crossfit_edge_value(
                torch.zeros(4, 3, 3),
                torch.zeros(5, 3),
                torch.zeros(4),
                torch.ones(4),
            )

    def test_negative_rss_is_refused(self):
        stats = list(_isotropic_fold_stats([[1.0, 0.0, 0.0]] * 4))
        stats[2][0] = -1.0
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            crossfit_edge_value(*stats)


if __name__ == "__main__":
    unittest.main()
