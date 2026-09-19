"""Per-vertex opacity field: the published model stays the default."""

import unittest


class OpacityFieldDefaultTest(unittest.TestCase):
    def test_training_default_is_the_min(self):
        from arguments import OptimizationParams

        class _Sink:
            def add_argument_group(self, _name):
                return self

            def add_argument(self, *_args, **_kwargs):
                return None

        self.assertFalse(OptimizationParams(_Sink()).opacity_field)

    def test_fresh_model_is_min_pooled(self):
        from scene.triangle_model import TriangleModel

        self.assertFalse(TriangleModel(3).opacity_field)

    def test_face_opacity_follows_the_model(self):
        """Pruning reads face_opacity, so it must be the min under the min model
        and the corner mean (the field's exact face average) under the field."""
        import torch
        from scene.triangle_model import TriangleModel

        model = TriangleModel(3)
        model.opacity_floor = 0.8
        corners = torch.tensor([[0.81], [0.88], [0.97], [0.99]])
        model.vertex_weight = model.inverse_opacity_activation(corners)
        model._triangle_indices = torch.tensor([[0, 1, 2], [1, 2, 3]])

        expected_min = torch.tensor([0.81, 0.88])
        expected_mean = torch.tensor([(0.81 + 0.88 + 0.97) / 3, (0.88 + 0.97 + 0.99) / 3])
        self.assertTrue(torch.allclose(model.face_opacity(), expected_min, atol=1e-5))
        model.opacity_field = True
        self.assertTrue(torch.allclose(model.face_opacity(), expected_mean, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
