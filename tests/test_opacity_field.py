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


if __name__ == "__main__":
    unittest.main()
