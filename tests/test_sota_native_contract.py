import unittest
from pathlib import Path


class SOTANativeContractTest(unittest.TestCase):
    def setUp(self):
        repo = Path(__file__).resolve().parents[1]
        self.environment = (repo / "sota" / "ensure_environment.sh").read_text(
            encoding="utf-8"
        )
        self.batch = (repo / "sota" / "batch9.sh").read_text(encoding="utf-8")
        self.spatial_batch = (repo / "sota" / "batch10.sh").read_text(encoding="utf-8")
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

    def test_nvcc_is_bound_to_the_pytorch_environment(self):
        self.assertIn("SOTA_CUDA_HOME=$(\"$SOTA_PYTHON\"", self.environment)
        self.assertIn('export CUDA_HOME=$SOTA_CUDA_HOME', self.environment)
        self.assertIn('export PATH="$CUDA_HOME/bin:$PATH"', self.environment)
        self.assertIn("nvcc/PyTorch CUDA mismatch", self.environment)

    def test_spatial_detail_pilot_is_one_locked_arm(self):
        self.assertIn('source "$HERE/ensure_environment.sh"', self.spatial_batch)
        self.assertIn('"$HERE/run.sh" tex2_sh2', self.spatial_batch)
        self.assertIn(
            "--max_points 2800000 --texel_order 2 --sh_degree 2",
            self.spatial_batch,
        )


if __name__ == "__main__":
    unittest.main()
