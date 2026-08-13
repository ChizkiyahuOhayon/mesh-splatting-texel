import json
from pathlib import Path
import tempfile
import unittest

from gorfe_v1_finalize import finalize


def _write(path: Path, text="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class FinalizePrepareTest(unittest.TestCase):
    def _fixture(self, root: Path):
        for name in (
            "build.log",
            "gpu.txt",
            "native_extension.so",
            "native_smoke.log",
            "python_env.txt",
            "tests.log",
        ):
            _write(root / name)
        _write(root / "native_result.json", '{"decision":"pass"}\n')
        for scene in ("garden", "room"):
            _write(
                root / f"{scene}_preflight.json",
                json.dumps(
                    {
                        "phase": "prepare-preflight",
                        "scene": scene,
                        "checks": {"valid": True},
                    }
                )
                + "\n",
            )
        _write(
            root
            / "wheels"
            / "diff_triangle_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl",
            "wheel",
        )
        for scene in ("garden", "room"):
            for name in ("candidate_manifest.json", "candidate_state.pt", "SHA256SUMS"):
                _write(root / scene / name)
            _write(
                root / scene / "result.json",
                json.dumps({"phase": "prepare", "decision": "prepared"}) + "\n",
            )
            _write(root / scene / "DONE", "complete\n")

    def test_complete_attempt_is_sealed_once_and_hashes_the_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            observed = finalize(root, phase="prepare")
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(observed["decision"], "prepared")
            self.assertEqual((root / "DONE").read_text(encoding="utf-8"), "complete\n")
            self.assertIn(
                "wheels/diff_triangle_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl",
                manifest["artifacts"],
            )
            with self.assertRaises(FileExistsError):
                finalize(root, phase="prepare")

    def test_invalid_child_attempt_cannot_be_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            _write(root / "garden" / "INVALID", "bad\n")
            with self.assertRaisesRegex(RuntimeError, "invalid attempt"):
                finalize(root, phase="prepare")


class FinalizeEvaluateTest(unittest.TestCase):
    def test_valid_scientific_failure_is_still_a_complete_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "gpu.txt",
                "install.log",
                "native_smoke.log",
                "python_env.txt",
                "tests.log",
                "decision_manifest.json",
                "DECISION_SHA256SUMS",
            ):
                _write(root / name)
            _write(root / "freeze_verify.json", '{"decision":"pass"}\n')
            _write(root / "extension_verify.json", '{"decision":"pass"}\n')
            _write(root / "native_result.json", '{"decision":"pass"}\n')
            _write(
                root / "decision.json",
                json.dumps({"phase": "overall", "decision": "fail"}) + "\n",
            )
            for scene in ("garden", "room"):
                for name in ("evaluation_state.pt", "manifest.json", "SHA256SUMS"):
                    _write(root / scene / name)
                _write(
                    root / scene / "result.json",
                    json.dumps({"phase": "evaluate", "scene": scene}) + "\n",
                )
                _write(root / scene / "DONE", "complete\n")
            observed = finalize(root, phase="evaluate")
            self.assertEqual(observed["decision"], "fail")
            self.assertEqual((root / "DONE").read_text(encoding="utf-8"), "complete\n")


if __name__ == "__main__":
    unittest.main()
