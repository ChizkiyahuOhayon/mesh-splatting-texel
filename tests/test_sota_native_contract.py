import unittest
from pathlib import Path


class SOTANativeContractTest(unittest.TestCase):
    def setUp(self):
        repo = Path(__file__).resolve().parents[1]
        self.environment = (repo / "sota" / "ensure_environment.sh").read_text(
            encoding="utf-8"
        )
        self.batch = (repo / "sota" / "batch9.sh").read_text(encoding="utf-8")
        self.runner = (repo / "sota" / "run.sh").read_text(encoding="utf-8")

    def test_stale_native_install_is_rebuilt(self):
        self.assertIn(
            "rev-parse HEAD:submodules/diff-triangle-mesh-rasterization",
            self.environment,
        )
        self.assertIn('"screen_space_gradients"', self.environment)
        self.assertIn('"sigma_face"', self.environment)
        self.assertIn("--force-reinstall", self.environment)

    def test_one_interpreter_is_used_for_preflight_and_training(self):
        self.assertIn('source "$HERE/ensure_environment.sh"', self.batch)
        self.assertIn('TRAIN_PYTHON=${MESH_SPLATTING_PYTHON:-python}', self.runner)
        self.assertIn('"$TRAIN_PYTHON" -u train.py', self.runner)


if __name__ == "__main__":
    unittest.main()
