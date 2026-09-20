"""Training-time OATS: the budget-matched pruning rule and the face buffer.

``train.py`` deletes faces whose peak ``max_blending`` sits at or below a rising
threshold, and samples densification from the same peak. Training-time OATS
swaps the statistic for the integral ``sum(alpha * T)`` without touching the
budget: the integral chooses *which* faces go, never how many. These tests pin
that contract and the alignment of the new per-face buffer, which desyncs
silently if a prune path forgets it.
"""

import unittest

import torch

from sota.survival import budget_matched_delete


class BudgetMatchedDeleteTest(unittest.TestCase):
    def test_deletes_exactly_the_peak_budget(self):
        peak_delete = torch.tensor([True, False, True, False, True])
        integrated = torch.tensor([9.0, 0.1, 8.0, 0.2, 7.0])
        delete = budget_matched_delete(peak_delete, integrated)
        self.assertEqual(int(delete.sum()), int(peak_delete.sum()))

    def test_the_integral_repicks_the_faces(self):
        """A face with one bright pixel and nothing else survives the peak rule;
        a face that is dim everywhere but always visible does not. The integral
        reverses both, which is the whole mechanism."""
        peak_delete = torch.tensor([False, True])
        integrated = torch.tensor([0.5, 40.0])
        delete = budget_matched_delete(peak_delete, integrated)
        self.assertTrue(bool(delete[0]) and not bool(delete[1]))

    def test_faces_that_never_contributed_go_first(self):
        peak_delete = torch.tensor([True, True, False, False])
        integrated = torch.tensor([3.0, 0.0, 0.0, 5.0])
        delete = budget_matched_delete(peak_delete, integrated)
        self.assertTrue(torch.equal(delete, torch.tensor([False, True, True, False])))

    def test_an_empty_budget_deletes_nothing(self):
        peak_delete = torch.zeros(4, dtype=torch.bool)
        delete = budget_matched_delete(peak_delete, torch.rand(4))
        self.assertEqual(int(delete.sum()), 0)

    def test_ties_break_by_face_index(self):
        peak_delete = torch.tensor([True, True, False, False])
        delete = budget_matched_delete(peak_delete, torch.ones(4))
        self.assertTrue(torch.equal(delete, torch.tensor([True, True, False, False])))

    def test_a_mismatched_score_vector_is_refused(self):
        with self.assertRaises(ValueError):
            budget_matched_delete(torch.zeros(4, dtype=torch.bool), torch.zeros(3))


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class IntegratedScoreBufferTest(unittest.TestCase):
    """The buffer is face-indexed, so every topology change must carry it."""

    def _model(self, faces=6):
        from scene.triangle_model import TriangleModel
        model = TriangleModel.__new__(TriangleModel)
        model._triangle_indices = torch.arange(faces * 3, dtype=torch.int32,
                                               device="cuda").reshape(faces, 3)
        model.image_size = torch.zeros(faces, device="cuda")
        model.importance_score = torch.zeros(faces, device="cuda")
        model.integrated_score = torch.arange(faces, dtype=torch.float32, device="cuda")
        model.pixel_count = torch.zeros(faces, dtype=torch.int, device="cuda")
        model.texel_order = 0
        model.face_hardness_enabled = False
        return model

    def test_pruning_carries_the_buffer(self):
        from scene.triangle_model import TriangleModel
        model = self._model()
        mask = torch.tensor([True, False, True, False, True, False], device="cuda")
        TriangleModel.prune_triangles(model, mask)
        self.assertTrue(torch.equal(model.integrated_score,
                                    torch.tensor([0.0, 2.0, 4.0], device="cuda")))

    def test_the_alignment_check_catches_a_stale_buffer(self):
        from scene.triangle_model import TriangleModel
        model = self._model()
        model.integrated_score = torch.zeros(3, device="cuda")
        with self.assertRaises(AssertionError):
            TriangleModel.validate_face_state(model)


class CombinedArgsTest(unittest.TestCase):
    """``get_combined_args`` merges the run's cfg_args over the command line and
    drops every argument whose value is None, so an optional flag that is left
    out has no attribute at all. survival_cleanup reads --budget that way."""

    def test_an_omitted_optional_argument_is_absent(self):
        from argparse import ArgumentParser
        from unittest import mock

        from arguments import get_combined_args

        parser = ArgumentParser()
        # model_path None sends get_combined_args down its "no cfg_args" path.
        parser.add_argument("--model_path", "-m", default=None)
        parser.add_argument("--budget", type=int, default=None)
        with mock.patch("sys.argv", ["prog"]):
            parsed = get_combined_args(parser)
        self.assertFalse(hasattr(parsed, "budget"))
        self.assertIsNone(getattr(parsed, "budget", None))


if __name__ == "__main__":
    unittest.main()
