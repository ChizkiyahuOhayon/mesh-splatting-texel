"""Pure-tensor tests for the visibility-aware terminal opacity (VATO) module."""

import unittest

import torch

from sota.visibility import (
    VisibilityTracker,
    activation_with_floor,
    adaptive_floor_schedule,
    check_adaptive_contract,
    endpoint_from_ambiguity,
    face_visibility_counts,
    inverse_activation_with_floor,
    pool_to_vertices,
    reparameterize_logits,
    shuffle_control,
)


def _v1_activation(raw, floor):
    """The frozen v1 expression, copied verbatim from scene/triangle_model.py."""
    return floor + (1.0 - floor) * torch.sigmoid(raw)


def _v1_inverse(y, floor, eps=1e-6):
    x = ((y.clamp(floor + eps, 1.0 - eps) - floor) / (1.0 - floor + eps))
    return torch.log(x / (1 - x))  # utils.general_utils.inverse_sigmoid, verbatim


class FaceCountTest(unittest.TestCase):
    def test_counts_match_hand_built_render(self):
        # 2 x 3 supersampled image, three faces; -1 is background.
        rend_ids = torch.tensor([[[0.0, 0.0, 1.0], [2.0, -1.0, 0.0]]])
        was_rendered = torch.tensor([4, 2, 1], dtype=torch.int32)
        dom, hit = face_visibility_counts(rend_ids, was_rendered, 3)
        torch.testing.assert_close(dom, torch.tensor([3.0, 1.0, 1.0]))
        torch.testing.assert_close(hit, torch.tensor([4.0, 2.0, 1.0]))

    def test_background_is_ignored_and_out_of_range_is_refused(self):
        rend_ids = torch.full((1, 2, 2), -1.0)
        dom, hit = face_visibility_counts(rend_ids, torch.zeros(2, dtype=torch.int32), 2)
        self.assertEqual(float(dom.sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "face id"):
            face_visibility_counts(torch.tensor([[[5.0]]]), torch.zeros(2, dtype=torch.int32), 2)
        with self.assertRaisesRegex(ValueError, "was_rendered"):
            face_visibility_counts(rend_ids, torch.zeros(1, dtype=torch.int32), 2)

    def test_dominance_never_exceeds_hits_when_counts_are_consistent(self):
        rend_ids = torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])
        was_rendered = torch.tensor([1, 3], dtype=torch.int32)
        dom, hit = face_visibility_counts(rend_ids, was_rendered, 2)
        self.assertTrue(bool((dom <= hit).all()))


class VertexPoolingTest(unittest.TestCase):
    def test_pooled_ratio_is_not_the_mean_of_ratios(self):
        # Vertex 0 is shared by a fully dominant face (dom 10 / hit 10) and a
        # never-dominant face (dom 0 / hit 30). Pooled ratio is 10/40 = 0.25,
        # whereas the mean of ratios would be 0.5.
        faces = torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.int32)
        dom = torch.tensor([10.0, 0.0])
        hit = torch.tensor([10.0, 30.0])
        dom_v, hit_v = pool_to_vertices(faces, dom, hit, 5)
        torch.testing.assert_close(dom_v, torch.tensor([10.0, 10.0, 10.0, 0.0, 0.0]))
        torch.testing.assert_close(hit_v, torch.tensor([40.0, 10.0, 10.0, 30.0, 30.0]))
        self.assertAlmostEqual(float(dom_v[0] / hit_v[0]), 0.25)

    def test_face_count_mismatch_is_refused(self):
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "faces"):
            pool_to_vertices(faces, torch.zeros(2), torch.zeros(2), 3)


class TrackerTest(unittest.TestCase):
    def test_ratio_is_bounded_and_unseen_vertices_are_neutral(self):
        tracker = VisibilityTracker(4, rho=0.5, device="cpu")
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        rend_ids = torch.tensor([[[0.0, 0.0, -1.0, -1.0]]])
        was_rendered = torch.tensor([4], dtype=torch.int32)
        tracker.update(rend_ids, was_rendered, faces)
        ratio = tracker.ratio()
        self.assertEqual(tuple(ratio.shape), (4,))
        self.assertTrue(bool((ratio >= 0).all() and (ratio <= 1).all()))
        torch.testing.assert_close(ratio[:3], torch.full((3,), 0.5))
        # A vertex that has never been hit is neutral (ratio one), so it keeps
        # the reference endpoint instead of being softened by default.
        self.assertEqual(float(ratio[3]), 1.0)
        self.assertEqual(tracker.steps, 1)

    def test_ema_forgets_old_observations(self):
        tracker = VisibilityTracker(3, rho=0.5, device="cpu")
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        dominant = torch.tensor([[[0.0, 0.0]]])
        hidden = torch.tensor([[[-1.0, -1.0]]])
        hits = torch.tensor([2], dtype=torch.int32)
        tracker.update(dominant, hits, faces)
        self.assertAlmostEqual(float(tracker.ratio()[0]), 1.0)
        for _ in range(20):
            tracker.update(hidden, hits, faces)
        self.assertLess(float(tracker.ratio()[0]), 1e-4)

    def test_prune_keeps_rows_aligned_and_mismatch_raises(self):
        tracker = VisibilityTracker(4, rho=0.9, device="cpu")
        tracker.dominant += torch.arange(4.0)
        tracker.hits += 1.0
        tracker.prune(torch.tensor([True, False, True, True]))
        torch.testing.assert_close(tracker.dominant, torch.tensor([0.0, 2.0, 3.0]))
        faces = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)[:, :3]
        with self.assertRaisesRegex(ValueError, "vertices"):
            tracker.update(torch.zeros(1, 1, 1), torch.zeros(1, dtype=torch.int32),
                           torch.tensor([[0, 1, 5]], dtype=torch.int32))
        del faces

    def test_invalid_rho_is_refused(self):
        with self.assertRaisesRegex(ValueError, "rho"):
            VisibilityTracker(1, rho=1.0, device="cpu")


class EndpointMappingTest(unittest.TestCase):
    def test_bounds_and_endpoints(self):
        d = torch.tensor([0.0, 0.5, 1.0, 1.5, -0.5])
        tau = endpoint_from_ambiguity(d, 0.6, 0.8)
        torch.testing.assert_close(tau, torch.tensor([0.6, 0.7, 0.8, 0.8, 0.6]))

    def test_equal_bounds_are_the_neutral_family(self):
        d = torch.rand(10)
        torch.testing.assert_close(endpoint_from_ambiguity(d, 0.8, 0.8), torch.full((10,), 0.8))

    def test_invalid_range_is_refused(self):
        with self.assertRaisesRegex(ValueError, "low"):
            endpoint_from_ambiguity(torch.zeros(1), 0.9, 0.8)

    def test_schedule_matches_the_v1_ramp_per_vertex(self):
        tau = torch.tensor([0.6, 0.8])
        # a = 0.5 of the way from init 0.1 to the endpoint.
        floor = adaptive_floor_schedule(tau, init_opacity=0.1, progress=0.5)
        torch.testing.assert_close(floor, torch.tensor([0.35, 0.45]))
        torch.testing.assert_close(adaptive_floor_schedule(tau, 0.1, 1.0), tau)
        torch.testing.assert_close(adaptive_floor_schedule(tau, 0.1, 7.0), tau)


class ShuffleControlTest(unittest.TestCase):
    def test_shuffle_preserves_multiset_and_changes_assignment(self):
        tau = torch.linspace(0.6, 0.8, 64)
        shuffled = shuffle_control(tau, seed=1234)
        torch.testing.assert_close(torch.sort(shuffled).values, torch.sort(tau).values)
        self.assertFalse(torch.equal(shuffled, tau))
        # Deterministic for provenance.
        torch.testing.assert_close(shuffle_control(tau, seed=1234), shuffled)


class FloorMathTest(unittest.TestCase):
    def test_scalar_floor_is_bitwise_the_v1_expression(self):
        raw = torch.randn(1000, 1)
        for floor in (0.1, 0.35, 0.8, 0.9999):
            self.assertTrue(torch.equal(activation_with_floor(raw, floor), _v1_activation(raw, floor)))
            y = _v1_activation(raw, floor)
            self.assertTrue(torch.equal(inverse_activation_with_floor(y, floor), _v1_inverse(y, floor)))

    def test_uniform_tensor_floor_equals_scalar_floor(self):
        raw = torch.randn(500, 1)
        floor = torch.full((500, 1), 0.8)
        tensor_path = activation_with_floor(raw, floor)
        scalar_path = activation_with_floor(raw, 0.8)
        torch.testing.assert_close(tensor_path, scalar_path, rtol=0.0, atol=1e-7)
        y = scalar_path
        torch.testing.assert_close(inverse_activation_with_floor(y, floor),
                                   inverse_activation_with_floor(y, 0.8), rtol=0.0, atol=1e-5)

    def test_per_vertex_floor_bounds_each_vertex_separately(self):
        raw = torch.tensor([[-50.0], [-50.0], [50.0]])
        floor = torch.tensor([[0.6], [0.8], [0.6]])
        o = activation_with_floor(raw, floor)
        torch.testing.assert_close(o, torch.tensor([[0.6], [0.8], [1.0]]))

    def test_reparameterization_preserves_realized_opacity_above_the_floor(self):
        raw = torch.randn(200, 1)
        old = 0.5
        realized = activation_with_floor(raw, old)
        new_floor = torch.full((200, 1), 0.45)  # lowered floor: nothing may change
        logits = reparameterize_logits(realized, new_floor)
        torch.testing.assert_close(activation_with_floor(logits, new_floor), realized, atol=1e-5, rtol=0.0)
        raised = torch.full((200, 1), 0.7)  # raised floor: clamp only the vertices below it
        logits = reparameterize_logits(realized, raised)
        out = activation_with_floor(logits, raised)
        expected = torch.maximum(realized, raised + 1e-6)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=0.0)

    def test_floor_shape_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "floor"):
            activation_with_floor(torch.zeros(3, 1), torch.zeros(2, 1))


class ContractTest(unittest.TestCase):
    def test_adaptive_checkpoint_contract(self):
        state = {
            "opacity_floor": 0.8,
            "opacity_floor_vertex": torch.tensor([[0.6], [0.7], [0.8]]),
            "adaptive_opacity": {"low": 0.6, "high": 0.8, "ema": 0.995, "control": "none"},
        }
        check_adaptive_contract(state, expect_adaptive=True, expected_high=0.8)
        with self.assertRaisesRegex(RuntimeError, "adaptive"):
            check_adaptive_contract(state, expect_adaptive=False, expected_high=0.8)
        bad = dict(state, opacity_floor_vertex=torch.tensor([[0.5]]))
        with self.assertRaisesRegex(RuntimeError, "range"):
            check_adaptive_contract(bad, expect_adaptive=True, expected_high=0.8)
        v1 = {"opacity_floor": 0.8}
        check_adaptive_contract(v1, expect_adaptive=False, expected_high=0.8)
        with self.assertRaisesRegex(RuntimeError, "adaptive"):
            check_adaptive_contract(v1, expect_adaptive=True, expected_high=0.8)


if __name__ == "__main__":
    unittest.main()


class RankStatisticTest(unittest.TestCase):
    def test_auc_extremes_and_ties(self):
        from sota.visibility import roc_auc
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc(scores, torch.tensor([False, False, True, True])), 1.0)
        self.assertAlmostEqual(roc_auc(scores, torch.tensor([True, True, False, False])), 0.0)
        self.assertAlmostEqual(roc_auc(torch.ones(6), torch.tensor([1, 0, 1, 0, 1, 0]).bool()), 0.5)
        self.assertAlmostEqual(roc_auc(scores, torch.zeros(4, dtype=torch.bool)), 0.5)

    def test_spearman_matches_known_values(self):
        from sota.visibility import spearman
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(spearman(x, x * 3 + 1), 1.0)
        self.assertAlmostEqual(spearman(x, -x), -1.0)
        self.assertAlmostEqual(spearman(x, torch.tensor([3.0, 1.0, 5.0, 2.0, 4.0])), 0.3, places=6)
        self.assertEqual(spearman(x, torch.ones(5)), 0.0)
