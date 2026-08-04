import unittest

import torch

from rits_prolongation import (
    FINETUNE_LRS,
    fd_probe_indices,
    fd_ratio_probe,
    install_trainable_split,
)
from rits_t0_decide import decide
from tests.test_rits_prolongation import DummyModel


class FdProbeTest(unittest.TestCase):
    def test_indices_pick_the_largest_magnitudes(self):
        gradient = torch.tensor([[0.1, -5.0, 0.0], [2.0, -0.3, 4.0]])
        indices = fd_probe_indices(gradient, count=3)
        self.assertEqual(set(indices.tolist()), {1, 3, 5})

    def test_count_is_clamped_to_the_gradient_size(self):
        self.assertEqual(fd_probe_indices(torch.ones(3), count=8).numel(), 3)

    def test_ratio_probe_recovers_a_scaled_analytic_gradient(self):
        # A quadratic whose true slope is 4x the reported analytic gradient:
        # the probe must report that scale, not hide or fail on it.
        parameter = torch.zeros(5)
        slope = torch.tensor([1.0, -3.0, 0.5, 2.0, 0.1])

        def loss():
            return float((slope * parameter).sum())

        result = fd_ratio_probe(parameter, slope / 4.0, loss, probes=3)
        self.assertAlmostEqual(result["median_ratio"], 4.0, places=3)
        self.assertLess(result["max_rung_disagreement"], 1e-6)
        self.assertEqual([row["index"] for row in result["rows"]][0], 1)

    def test_quadratic_probe_loss_converges_where_a_kinked_one_does_not(self):
        # The G0-lite instrument choice: a rendered image linear in the probed
        # scalar makes an MSE loss exactly quadratic, so both rungs agree; an
        # L1 loss kinks where the residual changes sign and they do not.
        parameter = torch.zeros(1)
        residual = torch.tensor([0.0005, -0.0011, 0.0016, -0.0004])

        def image():
            return residual + 0.282 * parameter

        quadratic = fd_ratio_probe(
            parameter, torch.ones(1), lambda: float(image().pow(2).mean()), probes=1
        )
        kinked = fd_ratio_probe(
            parameter, torch.ones(1), lambda: float(image().abs().mean()), probes=1
        )
        self.assertLess(quadratic["max_rung_disagreement"], 1e-6)
        self.assertGreater(kinked["max_rung_disagreement"], 0.05)

    def test_ratio_probe_restores_the_parameter(self):
        parameter = torch.tensor([0.25, -0.5])
        result = fd_ratio_probe(
            parameter, torch.tensor([1.0, 2.0]), lambda: float(parameter.sum()), probes=2
        )
        self.assertTrue(torch.equal(parameter, torch.tensor([0.25, -0.5])))
        self.assertIsNotNone(result["median_ratio"])


class InstallTrainableSplitTest(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.original = {
            "vertices": self.model.vertices.detach().clone(),
            "vertex_weight": self.model.vertex_weight.detach().clone(),
            "_features_dc": self.model._features_dc.detach().clone(),
            "_features_rest": self.model._features_rest.detach().clone(),
        }
        self.split = install_trainable_split(self.model, torch.tensor([0]))

    def test_counts_and_donors(self):
        self.assertEqual(self.split["base_face_count"], 2)
        self.assertEqual(self.split["split_face_count"], 5)
        self.assertEqual(self.split["child_face_start"], 1)
        self.assertEqual(
            self.split["split_vertex_count"],
            self.split["base_vertex_count"] + self.split["unique_edge_count"],
        )
        window_source, donor_indices, mode = self.split["window_donors"]
        self.assertEqual(mode, 7)
        self.assertTrue(torch.equal(window_source[:1], torch.tensor([-1], dtype=torch.int32)))
        self.assertTrue(torch.equal(window_source[1:], torch.zeros(4, dtype=torch.int32)))
        self.assertTrue(torch.equal(donor_indices, torch.tensor([[0, 1, 2]], dtype=torch.int32)))

    def test_whole_tensors_are_trainable_leaves_with_intact_prefix(self):
        base = self.split["base_vertex_count"]
        for name, original in self.original.items():
            tensor = getattr(self.model, name)
            self.assertTrue(tensor.is_leaf and tensor.requires_grad, name)
            self.assertTrue(torch.equal(tensor[:base].detach(), original), name)

    def test_optimizer_uses_restore_path_learning_rates(self):
        groups = {group["name"]: group for group in self.model.optimizer.param_groups}
        self.assertEqual(set(groups), set(FINETUNE_LRS))
        for name, expected in FINETUNE_LRS.items():
            self.assertEqual(groups[name]["lr"], expected)
            self.assertIs(
                groups[name]["params"][0],
                getattr(self.model, {"f_dc": "_features_dc", "f_rest": "_features_rest"}.get(name, name)),
            )


def _metrics(psnr, lpips):
    return {"psnr": psnr, "ssim": 0.9, "lpips_vgg": lpips}


def _table(garden, room):
    return {
        "garden": {arm: _metrics(*garden[arm]) for arm in ("unsplit", "abrupt", "rits")},
        "room": {arm: _metrics(*room[arm]) for arm in ("unsplit", "abrupt", "rits")},
    }


class DecideTest(unittest.TestCase):
    def test_clean_pass(self):
        verdict = decide(
            _table(
                {"unsplit": (25.0, 0.25), "abrupt": (25.1, 0.24), "rits": (25.3, 0.23)},
                {"unsplit": (28.0, 0.24), "abrupt": (28.05, 0.24), "rits": (28.2, 0.23)},
            )
        )
        self.assertTrue(verdict["pass"])
        self.assertTrue(all(verdict["checks"].values()))
        self.assertAlmostEqual(verdict["mean_psnr_gain_vs_unsplit"], 0.25)

    def test_losing_to_abrupt_on_one_scene_fails_the_causal_check(self):
        verdict = decide(
            _table(
                {"unsplit": (25.0, 0.25), "abrupt": (25.4, 0.23), "rits": (25.3, 0.23)},
                {"unsplit": (28.0, 0.24), "abrupt": (28.0, 0.24), "rits": (28.2, 0.23)},
            )
        )
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["checks"]["psnr_beats_abrupt_on_both_scenes"])

    def test_scene_regression_vs_unsplit_fails(self):
        verdict = decide(
            _table(
                {"unsplit": (25.0, 0.25), "abrupt": (25.1, 0.24), "rits": (25.6, 0.23)},
                {"unsplit": (28.0, 0.24), "abrupt": (27.8, 0.25), "rits": (27.9, 0.23)},
            )
        )
        self.assertFalse(verdict["pass"])
        self.assertFalse(
            verdict["checks"]["no_scene_below_unsplit_by_more_than_0p05"]
        )


if __name__ == "__main__":
    unittest.main()
