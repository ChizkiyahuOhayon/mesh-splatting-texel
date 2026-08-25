import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sota.formal_table import METRICS, SCENES
from sota.opacity_ablation import ARM, main


class OpacityAblationTest(unittest.TestCase):
    def test_candidate_is_compared_with_the_frozen_formal_stock(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            formal_root = base / "formal"
            output_root = base / "ablation"
            formal_root.mkdir()
            rows = {}
            for scene in SCENES:
                stock = {key: 2.0 for key in METRICS}
                rows[scene] = {"stock": stock}
                output = output_root / scene / ARM
                output.mkdir(parents=True)
                result = {
                    "scene": scene,
                    "arm": ARM,
                    "checkpoint_bytes": 1.0,
                    "triangles": 1.0,
                    "vertices": 1.0,
                    "metrics": {
                        "l1": 1.0,
                        "psnr": 3.0,
                        "ssim": 3.0,
                        "lpips_vgg": 1.0,
                        "fps": 3.0,
                    },
                }
                (output / "result.json").write_text(json.dumps(result))
            (formal_root / "formal_table.json").write_text(
                json.dumps({"rows": rows})
            )

            with redirect_stdout(io.StringIO()):
                main(formal_root, output_root)

            table = json.loads((output_root / "ablation_table.json").read_text())
            self.assertEqual(table["win_counts_vs_stock"]["psnr"], 9)
            self.assertEqual(table["delta_mean_vs_stock"]["fps"], 1.0)
            self.assertEqual((output_root / "DONE").read_text(), "complete\n")


if __name__ == "__main__":
    unittest.main()
