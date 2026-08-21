import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sota.main_table import main


class MainTableTest(unittest.TestCase):
    def test_directions_and_deltas_are_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene in ("garden", "room", "bicycle"):
                for arm, offset in (("stock", 0.0), ("ours", 1.0)):
                    output = root / scene / arm
                    output.mkdir(parents=True)
                    payload = {
                        "scene": scene,
                        "arm": arm,
                        "checkpoint_bytes": 100 - 10 * offset,
                        "triangles": 50 - 5 * offset,
                        "vertices": 30 - 3 * offset,
                        "metrics": {
                            "l1": 2.0 - offset,
                            "psnr": 2.0 + offset,
                            "ssim": 2.0 + offset,
                            "lpips_vgg": 2.0 - offset,
                            "fps": 2.0 + offset,
                        },
                    }
                    (output / "result.json").write_text(json.dumps(payload))
            with redirect_stdout(io.StringIO()):
                main(root)
            table = json.loads((root / "main_table.json").read_text())
            self.assertEqual(table["win_counts"]["psnr"], 3)
            self.assertEqual(
                table["rows"]["room"]["delta_ours_minus_stock"]["fps"],
                1.0,
            )
            self.assertEqual((root / "DONE").read_text(), "complete\n")


if __name__ == "__main__":
    unittest.main()
