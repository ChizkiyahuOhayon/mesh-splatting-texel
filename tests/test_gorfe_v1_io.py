import json
from pathlib import Path
import tempfile
import unittest

import torch

from gorfe_v1_io import (
    artifact_records,
    directory_bytes,
    save_torch_new,
    sha256_file,
    write_json_new,
    write_sha256s,
)


class WriteOnceArtifactTest(unittest.TestCase):
    def test_json_and_tensor_refuse_overwrite_and_are_hashable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = write_json_new(root / "record.json", {"z": 2, "a": 1})
            state_path = save_torch_new(root / "state.pt", {"x": torch.arange(3)})
            self.assertEqual(json.loads(json_path.read_text()), {"a": 1, "z": 2})
            self.assertEqual(
                torch.load(state_path, weights_only=True)["x"].tolist(), [0, 1, 2]
            )
            with self.assertRaises(FileExistsError):
                write_json_new(json_path, {})
            with self.assertRaises(FileExistsError):
                save_torch_new(state_path, {})
            records = artifact_records({"json": json_path, "state": state_path})
            self.assertEqual(records["json"]["sha256"], sha256_file(json_path))
            self.assertGreaterEqual(directory_bytes(root), json_path.stat().st_size)

    def test_checksum_file_is_itself_write_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = write_json_new(root / "result.json", {"ok": True})
            sums = write_sha256s(root / "SHA256SUMS", [artifact])
            self.assertIn(sha256_file(artifact), sums.read_text())
            with self.assertRaises(FileExistsError):
                write_sha256s(sums, [artifact])


if __name__ == "__main__":
    unittest.main()
