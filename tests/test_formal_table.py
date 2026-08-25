import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sota.formal_table import ARMS, SCENES, main


class FormalTableTest(unittest.TestCase):
    def test_three_arms_are_aggregated_over_all_nine_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene in SCENES:
                for index, arm in enumerate(ARMS):
                    output = root / scene / arm
                    output.mkdir(parents=True)
                    result = {
                        "scene": scene,
                        "arm": arm,
                        "checkpoint_bytes": 100 - index,
                        "triangles": 90 - index,
                        "vertices": 80 - index,
                        "metrics": {
                            "l1": 3.0 - index,
                            "psnr": 3.0 + index,
                            "ssim": 3.0 + index,
                            "lpips_vgg": 3.0 - index,
                            "fps": 3.0 + index,
                        },
                    }
                    (output / "result.json").write_text(json.dumps(result))
            with redirect_stdout(io.StringIO()):
                main(root)
            table = json.loads((root / "formal_table.json").read_text())
            self.assertEqual(table["win_counts_vs_stock"]["ours_speed"]["psnr"], 9)
            self.assertEqual(table["delta_mean_vs_stock"]["ours_quality"]["fps"], 2.0)
            self.assertEqual((root / "DONE").read_text(), "complete\n")


if __name__ == "__main__":
    unittest.main()
