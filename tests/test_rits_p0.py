import unittest

import torch

from rits_p0_decide import REQUIRED_REDUCTION, decide
from rits_prolongation import (
    install_trainable_split,
    original_prefix_unchanged,
    zero_original_gradients,
)
from tests.test_rits_prolongation import DummyModel


class FreezeOriginalRowsTest(unittest.TestCase):
    """An Adam step must move only the rows the split appended."""

    def setUp(self):
        self.model = DummyModel()
        self.originals = {
            name: getattr(self.model, name).detach().clone()
            for name in ("vertices", "vertex_weight", "_features_dc", "_features_rest")
        }
        self.split = install_trainable_split(self.model, torch.tensor([0]))
        self.base = self.split["base_vertex_count"]

    def _step(self, zero_originals):
        loss = (
            self.model.vertices.pow(2).sum()
            + self.model._features_dc.pow(2).sum()
            + self.model._features_rest.pow(2).sum()
        )
        loss.backward()
        if zero_originals:
            zero_original_gradients(self.model, self.base)
        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)

    def test_zeroed_prefix_survives_an_optimizer_step(self):
        self._step(zero_originals=True)
        self.assertTrue(original_prefix_unchanged(self.model, self.originals))

    def test_midpoint_rows_still_move(self):
        before = self.model._features_dc[self.base :].detach().clone()
        self._step(zero_originals=True)
        self.assertFalse(torch.equal(self.model._features_dc[self.base :].detach(), before))

    def test_the_check_catches_an_unfrozen_prefix(self):
        self._step(zero_originals=False)
        self.assertFalse(original_prefix_unchanged(self.model, self.originals))


class DecideTest(unittest.TestCase):
    def test_an_80_percent_reduction_passes(self):
        checks = decide(
            {"probe_region_mae": 4.2e-3, "global_mae": 1.1e-4},
            {"probe_region_mae": 4.2e-3 * (1.0 - REQUIRED_REDUCTION), "global_mae": 2.0e-5},
        )
        self.assertTrue(all(checks.values()))

    def test_a_shortfall_fails(self):
        checks = decide(
            {"probe_region_mae": 4.2e-3, "global_mae": 1.1e-4},
            {"probe_region_mae": 1.0e-3, "global_mae": 2.0e-5},
        )
        self.assertFalse(checks["probe_region_mae_reduced_by_80_percent"])

    def test_probe_gain_paid_for_with_a_worse_global_error_fails(self):
        checks = decide(
            {"probe_region_mae": 4.2e-3, "global_mae": 1.1e-4},
            {"probe_region_mae": 1.0e-5, "global_mae": 5.0e-4},
        )
        self.assertTrue(checks["probe_region_mae_reduced_by_80_percent"])
        self.assertFalse(checks["global_mae_improved"])


if __name__ == "__main__":
    unittest.main()
