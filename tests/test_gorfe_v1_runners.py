from pathlib import Path
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


class RunnerOrderingTest(unittest.TestCase):
    def test_evaluation_verifies_freeze_before_install_or_native_import(self):
        script = (
            REPOSITORY / "experiments/gorfe_v1/run_evaluate_gpu3.sh"
        ).read_text(encoding="utf-8")
        freeze = script.index('"${PYTHON_BIN}" gorfe_v1_freeze_verify.py')
        install = script.index('"${PYTHON_BIN}" -m pip install')
        native = script.index("from diff_triangle_rasterization import _C")
        scene = script.index('"${PYTHON_BIN}" gorfe_v1_evaluate.py')
        self.assertLess(freeze, install)
        self.assertLess(install, native)
        self.assertLess(native, scene)

    def test_preparation_and_evaluation_remain_separate_invocations(self):
        prepare = (
            REPOSITORY / "experiments/gorfe_v1/run_prepare_gpu3.sh"
        ).read_text(encoding="utf-8")
        evaluate = (
            REPOSITORY / "experiments/gorfe_v1/run_evaluate_gpu3.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("gorfe_v1_evaluate.py", prepare)
        self.assertNotIn("gorfe_v1_prepare.py", evaluate)

    def test_prepare_preflights_both_scenes_before_persistent_attempt(self):
        script = (
            REPOSITORY / "experiments/gorfe_v1/run_prepare_gpu3.sh"
        ).read_text(encoding="utf-8")
        garden = script.index("--scene garden")
        room = script.index("--scene room")
        reserve = script.index('mkdir "${OUT}"')
        build = script.index('"${PYTHON_BIN}" -m pip wheel')
        self.assertLess(garden, reserve)
        self.assertLess(room, reserve)
        self.assertLess(reserve, build)


if __name__ == "__main__":
    unittest.main()
