import unittest

import torch

from adc_allocator import leaf_cost
from adc_densify import (
    _remap_after_prune,
    _subdivide,
    eligible_probs,
    install_arm,
    install_multiplicity,
    install_rng_fix,
)


class FakeModel:
    """A mesh that behaves like TriangleModel's subdivision bookkeeping.

    `_update_params_fast` hands its payload straight to `densification_postfix`,
    which appends four children per selected face; `prune_triangles` takes a keep
    mask. Every face carries a lineage tag so a test can count how many leaves an
    original face ended up with -- the only way to tell depth two from depth one.
    """

    def __init__(self, faces=12, vertices_per_face=3, add_percentage=2.0):
        self._triangle_indices = torch.zeros(faces, 3, dtype=torch.int32)
        self.origin = torch.arange(faces)
        self.vertices = torch.zeros(faces * vertices_per_face, 3)
        self.importance_score = torch.ones(faces)
        self.image_size = torch.ones(faces)
        self.areas = torch.ones(faces)
        self.add_percentage = add_percentage
        self.size_probs_zero = 0.0
        self.size_probs_zero_image_space = 0.0
        self.subdivisions = []

    def triangle_areas(self):
        return self.areas

    def _update_params_fast(self, indices, iteration):
        self.subdivisions.append(indices.clone())
        return (indices,)

    def densification_postfix(self, indices):
        count = indices.numel()
        self._triangle_indices = torch.cat(
            [self._triangle_indices, torch.zeros(4 * count, 3, dtype=torch.int32)]
        )
        self.origin = torch.cat([self.origin, self.origin[indices].repeat_interleave(4)])
        self.areas = torch.cat([self.areas, (self.areas[indices] / 4).repeat_interleave(4)])
        self.importance_score = torch.cat([self.importance_score, torch.ones(4 * count)])
        self.image_size = torch.cat([self.image_size, torch.ones(4 * count)])

    def prune_triangles(self, keep):
        self._triangle_indices = self._triangle_indices[keep]
        self.origin = self.origin[keep]
        self.areas = self.areas[keep]
        self.importance_score = self.importance_score[keep]
        self.image_size = self.image_size[keep]

    def leaves_per_origin(self):
        return torch.bincount(self.origin, minlength=int(self.origin.max()) + 1)


class EligibleProbsTest(unittest.TestCase):
    """The published zeroing, unchanged: ADC-G0 alters no score, because XVR-G0
    already retired residual-guided selection with `max_blending` as a control."""

    def test_scores_pass_through_untouched(self):
        model = FakeModel(faces=4)
        model.importance_score = torch.tensor([0.1, 0.7, 0.2, 0.4])
        self.assertTrue(torch.equal(eligible_probs(model), model.importance_score))

    def test_faces_below_the_area_floor_are_zeroed(self):
        model = FakeModel(faces=3)
        model.areas = torch.tensor([1.0, 0.001, 1.0])
        model.size_probs_zero = 0.01
        self.assertEqual(eligible_probs(model).tolist(), [1.0, 0.0, 1.0])

    def test_faces_below_the_image_size_floor_are_zeroed(self):
        model = FakeModel(faces=3)
        model.image_size = torch.tensor([20.0, 1.0, 20.0])
        model.size_probs_zero_image_space = 10.0
        self.assertEqual(eligible_probs(model).tolist(), [1.0, 0.0, 1.0])

    def test_the_model_score_is_not_mutated(self):
        model = FakeModel(faces=3)
        model.areas = torch.zeros(3)
        model.size_probs_zero = 1.0
        eligible_probs(model)
        self.assertEqual(model.importance_score.tolist(), [1.0, 1.0, 1.0])


class RemapTest(unittest.TestCase):
    def test_survivors_move_down_by_the_removals_below_them(self):
        removed = torch.tensor([False, True, False, True, False])
        self.assertEqual(_remap_after_prune(torch.tensor([0, 2, 4]), removed).tolist(), [0, 1, 2])

    def test_nothing_moves_when_nothing_is_removed(self):
        removed = torch.zeros(4, dtype=torch.bool)
        self.assertEqual(_remap_after_prune(torch.tensor([1, 3]), removed).tolist(), [1, 3])


class SubdivideTest(unittest.TestCase):
    def test_children_land_where_the_returned_range_says(self):
        model = FakeModel(faces=6)
        children, _ = _subdivide(model, torch.tensor([1, 4]), iteration=0)
        self.assertEqual(model._triangle_indices.shape[0], 6 - 2 + 8)
        self.assertEqual(
            sorted(model.origin[children].tolist()), [1, 1, 1, 1, 4, 4, 4, 4]
        )

    def test_the_parents_are_gone(self):
        model = FakeModel(faces=5)
        _subdivide(model, torch.tensor([2]), iteration=0)
        self.assertEqual(int((model.origin == 2).sum()), 4)
        self.assertEqual(model._triangle_indices.shape[0], 8)


def _install_and_densify(model, seed=0, largest=0, iteration=1000):
    rounds = install_multiplicity(model, seed=seed)
    before = model._triangle_indices.shape[0]
    model.add_new_gs(iteration, cap_max=10**9, splitt_large_triangles=largest)
    return rounds, before


class MultiplicityTest(unittest.TestCase):
    def test_faces_created_equal_the_budget_actually_spent(self):
        """The invariant that makes the arms comparable at matched primitive count --
        the confound that invalidated SAC-G0."""
        model = FakeModel(faces=40)
        rounds, before = _install_and_densify(model)
        grew = model._triangle_indices.shape[0] - before
        self.assertEqual(grew, rounds[0]["spent"])

    def test_the_budget_is_never_overspent(self):
        model = FakeModel(faces=40)
        rounds, _ = _install_and_densify(model)
        self.assertLessEqual(rounds[0]["spent"], rounds[0]["budget"])

    def test_a_dominant_face_receives_sixteen_leaves(self):
        """Depth two is the whole point: the published allocator cannot give any face
        more than four leaves in a round, however it scored."""
        model = FakeModel(faces=30)
        model.importance_score = torch.full((30,), 1e-9)
        model.importance_score[7] = 1.0
        _install_and_densify(model)
        self.assertEqual(int(model.leaves_per_origin()[7]), 16)

    def test_a_flat_score_still_subdivides_many_faces(self):
        model = FakeModel(faces=40)
        rounds, _ = _install_and_densify(model)
        self.assertGreater(rounds[0]["depth_1"] + rounds[0]["depth_2"], 1)

    def test_the_histogram_is_the_manipulation_check(self):
        model = FakeModel(faces=30)
        model.importance_score = torch.full((30,), 1e-9)
        model.importance_score[3] = 1.0
        rounds, _ = _install_and_densify(model)
        self.assertGreaterEqual(rounds[0]["depth_2"], 1)
        self.assertEqual(
            rounds[0]["depth_0"] + rounds[0]["depth_1"] + rounds[0]["depth_2"], 30
        )

    def test_zero_probability_faces_are_never_subdivided(self):
        model = FakeModel(faces=20)
        model.importance_score = torch.zeros(20)
        model.importance_score[5:10] = 1.0
        _install_and_densify(model)
        leaves = model.leaves_per_origin()
        self.assertTrue(bool((leaves[:5] == 1).all()))
        self.assertTrue(bool((leaves[10:] == 1).all()))

    def test_the_largest_faces_keep_their_unconditional_split(self):
        """The published `topk(areas)` channel is present in both arms, so it is not
        part of what the gate tests."""
        model = FakeModel(faces=20)
        model.importance_score = torch.zeros(20)
        model.areas = torch.arange(1.0, 21.0)
        _install_and_densify(model, largest=3)
        self.assertTrue(bool((model.leaves_per_origin()[17:] >= 4).all()))

    def test_nothing_happens_when_no_budget_is_requested(self):
        model = FakeModel(faces=10, add_percentage=1.0)
        rounds, before = _install_and_densify(model)
        self.assertEqual(rounds, [])
        self.assertEqual(model._triangle_indices.shape[0], before)

    def test_it_is_reproducible_from_the_seed(self):
        results = []
        for _ in range(2):
            model = FakeModel(faces=40)
            _install_and_densify(model, seed=5)
            results.append(model.leaves_per_origin())
        self.assertTrue(torch.equal(*results))

    def test_successive_rounds_are_not_correlated(self):
        model = FakeModel(faces=60)
        rounds = install_multiplicity(model, seed=0)
        model.add_new_gs(1000, cap_max=10**9, splitt_large_triangles=0)
        first = model.leaves_per_origin().clone()
        model.add_new_gs(2000, cap_max=10**9, splitt_large_triangles=0)
        second = model.leaves_per_origin() - first
        self.assertEqual(len(rounds), 2)
        self.assertFalse(torch.equal(first, second))


class RngFixTest(unittest.TestCase):
    def test_the_ambient_stream_is_left_alone(self):
        model = FakeModel(faces=10)
        install_rng_fix(model, seed=0)
        torch.manual_seed(42)
        expected = torch.rand(3)
        torch.manual_seed(42)
        model._sample_alives(torch.ones(10), 4)
        self.assertTrue(torch.equal(torch.rand(3), expected))

    def test_it_still_samples_without_replacement(self):
        model = FakeModel(faces=10)
        install_rng_fix(model, seed=0)
        sampled = model._sample_alives(torch.ones(10), 6)
        self.assertEqual(sampled.numel(), 6)
        self.assertEqual(torch.unique(sampled).numel(), 6)

    def test_alive_indices_are_honoured(self):
        model = FakeModel(faces=10)
        install_rng_fix(model, seed=0)
        alive = torch.arange(50, 60)
        self.assertGreaterEqual(int(model._sample_alives(torch.ones(10), 4, alive).min()), 50)

    def test_successive_calls_differ(self):
        model = FakeModel(faces=200)
        install_rng_fix(model, seed=0)
        first = model._sample_alives(torch.ones(200), 20)
        second = model._sample_alives(torch.ones(200), 20)
        self.assertFalse(torch.equal(first, second))


class InstallArmTest(unittest.TestCase):
    def test_stock_installs_nothing(self):
        model = FakeModel(faces=10)
        install_arm(model, "stock", seed=0)
        self.assertFalse(hasattr(model, "add_new_gs"))
        self.assertFalse("_sample_alives" in vars(model))

    def test_named_arms_install(self):
        for arm in ("rng", "multiplicity"):
            model = FakeModel(faces=10)
            install_arm(model, arm, seed=0)
            self.assertIn("_sample_alives", vars(model))

    def test_an_unknown_arm_is_refused(self):
        with self.assertRaises(ValueError):
            install_arm(FakeModel(), "absgrad", seed=0)


if __name__ == "__main__":
    unittest.main()
