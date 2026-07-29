import json
import tempfile
import unittest
from pathlib import Path

from svsr_decide import decide


class SVSRDecisionTest(unittest.TestCase):
    def _write(self, root, scene, sh, fixed, footprint, confirmatory=True):
        root.mkdir()
        summary = {
            "sh": {"psnr": sh[0], "lpips_vgg": sh[1]},
            "fixed": {"psnr": fixed[0], "lpips_vgg": fixed[1]},
            "footprint": {"psnr": footprint[0], "lpips_vgg": footprint[1]},
        }
        (root / "results.json").write_text(json.dumps({"scene": scene, "summary": summary}))
        (root / "svsr_manifest.json").write_text(json.dumps({
            "confirmatory_settings": confirmatory,
        }))

    def test_passes_locked_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._write(tmp / "garden", "garden", (20.0, 0.40), (21.0, 0.30), (20.8, 0.32))
            self._write(tmp / "room", "room", (21.0, 0.30), (20.0, 0.35), (20.6, 0.34))
            self._write(tmp / "stump", "stump", (21.0, 0.30), (20.0, 0.35), (20.5, 0.35))
            self.assertEqual(decide([tmp / scene for scene in ("garden", "room", "stump")])["verdict"], "PASS")

    def test_smoke_cannot_declare_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._write(tmp / "garden", "garden", (20.0, 0.40), (21.0, 0.30), (20.8, 0.32), False)
            self._write(tmp / "room", "room", (21.0, 0.30), (20.0, 0.35), (20.6, 0.34), False)
            self._write(tmp / "stump", "stump", (21.0, 0.30), (20.0, 0.35), (20.5, 0.35), False)
            self.assertEqual(decide([tmp / scene for scene in ("garden", "room", "stump")])["verdict"], "EXPLORATORY_PASS")


if __name__ == "__main__":
    unittest.main()
