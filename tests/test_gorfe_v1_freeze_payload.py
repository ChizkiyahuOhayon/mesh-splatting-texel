import json
from pathlib import Path
import tempfile
import unittest

from gorfe_v1_freeze_payload import (
    ROOT_ARTIFACTS,
    SCENES,
    SCENE_ARTIFACTS,
    build_payload,
    preparation_artifact_paths,
)


class FreezePayloadTest(unittest.TestCase):
    def _fixture(self, root):
        for name in ROOT_ARTIFACTS:
            if name == "native_result.json":
                value = '{"decision":"pass"}\n'
            elif name == "manifest.json":
                value = json.dumps(
                    {
                        "schema": "gorfe-v1-phase-root-v1",
                        "phase": "prepare",
                        "decision": "prepared",
                        "source_revision": "abc123",
                    }
                ) + "\n"
            elif name == "DONE":
                value = "complete\n"
            else:
                value = "x\n"
            (root / name).write_text(value, encoding="utf-8")
        wheels = root / "wheels"
        wheels.mkdir()
        (wheels / "diff_triangle_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl").write_bytes(
            b"wheel"
        )
        for scene in SCENES:
            directory = root / scene
            directory.mkdir()
            for name in SCENE_ARTIFACTS:
                if name == "result.json":
                    value = json.dumps({"phase": "prepare", "decision": "prepared"}) + "\n"
                elif name == "DONE":
                    value = "complete\n"
                else:
                    value = f"{scene}-{name}\n"
                (directory / name).write_text(value, encoding="utf-8")

    def test_payload_binds_every_required_artifact_without_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            payload = build_payload(root, revision="abc123")
            expected = preparation_artifact_paths(root)
            self.assertEqual(
                set(payload["preparation_artifacts"]), set(expected)
            )
            self.assertEqual(payload["source_revision"], "abc123")
            for identity in payload["preparation_artifacts"].values():
                self.assertEqual(set(identity), {"bytes", "sha256"})

    def test_nonpassing_native_or_scene_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            (root / "native_result.json").write_text(
                '{"decision":"fail"}\n', encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                build_payload(root, revision="abc123")

    def test_ambiguous_frozen_wheel_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            (root / "wheels" / "diff_triangle_rasterization-0.0.1-cp311-cp311-linux_x86_64.whl").write_bytes(
                b"other"
            )
            with self.assertRaisesRegex(RuntimeError, "exactly one frozen"):
                build_payload(root, revision="abc123")


if __name__ == "__main__":
    unittest.main()
