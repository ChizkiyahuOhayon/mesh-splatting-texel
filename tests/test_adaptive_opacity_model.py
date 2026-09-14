"""TriangleModel contract for the per-vertex (VATO) opacity floor.

These tests import the real model, which needs the native extensions
(simple_knn, rdel). They are skipped where those are absent and must be run on
the A40 before any training with --adaptive_opacity.
"""

import tempfile
import unittest

import torch

try:
    from scene.triangle_model import TriangleModel
    from arguments import OptimizationParams
    MODEL_AVAILABLE = True
except Exception:  # noqa: BLE001 - native extensions absent on the laptop
    MODEL_AVAILABLE = False


class _ArgumentSink:
    def add_argument_group(self, _name):
        return self

    def add_argument(self, *_args, **_kwargs):
        return None


def _v1_activation(raw, floor):
    return floor + (1.0 - floor) * torch.sigmoid(raw)


def _small_model(n_vertices=12, n_faces=8, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = TriangleModel(3)
    model.vertices = torch.nn.Parameter(torch.randn(n_vertices, 3, device=device))
    model.vertex_weight = torch.nn.Parameter(torch.randn(n_vertices, 1, device=device))
    model._features_dc = torch.nn.Parameter(torch.randn(n_vertices, 1, 3, device=device))
    model._features_rest = torch.nn.Parameter(torch.randn(n_vertices, 15, 3, device=device))
    faces = torch.stack([torch.randperm(n_vertices, device=device)[:3] for _ in range(n_faces)])
    model._triangle_indices = faces.to(torch.int32)
    model._sigma = model.inverse_exponential_activation(1.0)
    model.image_size = torch.zeros(n_faces, device=device)
    model.importance_score = torch.zeros(n_faces, device=device)
    model.pixel_count = torch.zeros(n_faces, dtype=torch.int, device=device)
    model.training_setup(OptimizationParams(_ArgumentSink()), 0.001, 0.03, 0.001)
    return model


@unittest.skipUnless(MODEL_AVAILABLE, "native extensions required (run on the A40)")
class AdaptiveOpacityModelTest(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _small_model(device=self.device)

    def test_neutral_model_has_no_per_vertex_floor_and_matches_v1(self):
        model = self.model
        self.assertIsNone(model.opacity_floor_vertex)
        self.assertIsNone(model.adaptive_opacity)
        model.update_min_weight(0.5)
        self.assertIsNone(model.opacity_floor_vertex)
        self.assertEqual(model.opacity_floor, 0.5)
        raw = model.vertex_weight.detach()
        self.assertTrue(torch.equal(model.get_vertex_weight, _v1_activation(raw, 0.5)))
        # Subset calls keep working on the scalar path (train.py pruning block).
        subset = model.opacity_activation(model.vertex_weight[model._triangle_indices])
        self.assertEqual(tuple(subset.shape), (8, 3, 1))

    def test_tensor_floor_bounds_vertices_and_keeps_the_scalar_reference(self):
        model = self.model
        model.update_min_weight(0.5)
        before = model.get_vertex_weight.detach().clone()
        floors = torch.full((12, 1), 0.45, device=self.device)
        floors[:4] = 0.7
        model.update_min_weight(floors, reference_floor=0.8)
        self.assertEqual(model.opacity_floor, 0.8)
        self.assertIsNotNone(model.opacity_floor_vertex)
        after = model.get_vertex_weight.detach()
        # Lowered floor: unchanged. Raised floor: clamped only where below it.
        torch.testing.assert_close(after[4:], before[4:], atol=1e-5, rtol=0.0)
        torch.testing.assert_close(after[:4], torch.maximum(before[:4], floors[:4] + model.eps),
                                   atol=1e-5, rtol=0.0)
        self.assertTrue(bool((after >= model.opacity_floor_vertex).all()))

    def test_uniform_tensor_floor_reproduces_the_scalar_path(self):
        model = self.model
        raw = model.vertex_weight.detach().clone()
        scalar = _v1_activation(raw, 0.8)
        model.update_min_weight(torch.full((12, 1), 0.8, device=self.device), reference_floor=0.8)
        # update_min_weight preserves realized opacity, so compare activations of
        # the same logits under both floor representations instead.
        model.vertex_weight.data.copy_(raw)
        torch.testing.assert_close(model.get_vertex_weight, scalar, atol=1e-7, rtol=0.0)

    def test_subset_activation_needs_an_index_under_a_tensor_floor(self):
        model = self.model
        model.update_min_weight(torch.full((12, 1), 0.6, device=self.device), reference_floor=0.8)
        with self.assertRaisesRegex(ValueError, "floor"):
            model.opacity_activation(model.vertex_weight[:3])
        indexed = model.opacity_activation(model.vertex_weight[:3], index=torch.arange(3, device=self.device))
        torch.testing.assert_close(indexed, model.get_vertex_weight[:3])

    def test_scalar_update_after_tensor_floor_returns_to_the_global_path(self):
        model = self.model
        model.update_min_weight(torch.full((12, 1), 0.6, device=self.device), reference_floor=0.8)
        realized = model.get_vertex_weight.detach().clone()
        model.update_min_weight(0.6)
        self.assertIsNone(model.opacity_floor_vertex)
        torch.testing.assert_close(model.get_vertex_weight, realized, atol=1e-5, rtol=0.0)

    def test_vertex_prune_keeps_floor_and_dominance_aligned(self):
        model = self.model
        floors = torch.linspace(0.6, 0.8, 12, device=self.device).unsqueeze(1)
        model.update_min_weight(floors, reference_floor=0.8)
        model.visibility_dominance = torch.arange(12.0, device=self.device)
        mask = torch.ones(12, dtype=torch.bool, device=self.device)
        mask[[1, 5]] = False
        model._prune_vertex_optimizer(mask)
        self.assertEqual(model.opacity_floor_vertex.shape[0], 10)
        torch.testing.assert_close(model.opacity_floor_vertex, floors[mask])
        torch.testing.assert_close(model.visibility_dominance, torch.arange(12.0, device=self.device)[mask])
        model.validate_vertex_state()
        model.visibility_dominance = torch.zeros(3, device=self.device)
        with self.assertRaisesRegex(AssertionError, "visibility_dominance"):
            model.validate_vertex_state()

    def test_checkpoint_round_trip(self):
        model = self.model
        floors = torch.linspace(0.6, 0.8, 12, device=self.device).unsqueeze(1)
        model.update_min_weight(floors, reference_floor=0.8)
        model.adaptive_opacity = {"low": 0.6, "high": 0.8, "ema": 0.995, "control": "none"}
        model.visibility_dominance = torch.rand(12, device=self.device)
        with tempfile.TemporaryDirectory() as directory:
            model.save_parameters(directory)
            state = torch.load(f"{directory}/point_cloud_state_dict.pt", map_location="cpu")
            self.assertEqual(state["opacity_floor"], 0.8)
            self.assertIn("opacity_floor_vertex", state)
            self.assertEqual(state["adaptive_opacity"]["low"], 0.6)
            restored = TriangleModel(3)
            restored.load_parameters(directory, device=self.device)
            torch.testing.assert_close(restored.opacity_floor_vertex, floors)
            torch.testing.assert_close(restored.visibility_dominance, model.visibility_dominance)
            self.assertEqual(restored.adaptive_opacity["control"], "none")
            torch.testing.assert_close(restored.get_vertex_weight, model.get_vertex_weight.detach())

    def test_v1_checkpoint_loads_without_a_per_vertex_floor(self):
        model = self.model
        model.update_min_weight(0.8)
        with tempfile.TemporaryDirectory() as directory:
            model.save_parameters(directory)
            state = torch.load(f"{directory}/point_cloud_state_dict.pt", map_location="cpu")
            self.assertNotIn("opacity_floor_vertex", state)
            self.assertNotIn("adaptive_opacity", state)
            restored = TriangleModel(3)
            restored.load_parameters(directory, device=self.device)
            self.assertIsNone(restored.opacity_floor_vertex)
            self.assertIsNone(restored.adaptive_opacity)
            self.assertEqual(restored.opacity_floor, 0.8)


if __name__ == "__main__":
    unittest.main()
