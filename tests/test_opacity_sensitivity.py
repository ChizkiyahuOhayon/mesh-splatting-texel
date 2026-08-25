import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sota.formal_table import METRICS
from sota.opacity_sensitivity import SCENES, main


class OpacitySensitivityTest(unittest.TestCase):
    def test_trained_floors_are_combined_with_existing_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_root = root / "formal"
            opacity_root = root / "opacity"
            output_root = root / "sensitivity"
            formal_root.mkdir()
            opacity_root.mkdir()
            formal_rows = {}
            opacity_rows = {}
            for scene in SCENES:
                stock = {key: 1.0 for key in METRICS}
                opacity08 = stock | {"psnr": 4.0}
                formal_rows[scene] = {"stock": stock}
                opacity_rows[scene] = {"ours_opacity": opacity08}
                for arm, psnr in (("ours_opacity07", 2.0), ("ours_opacity09", 3.0)):
                    output = output_root / scene / arm
                    output.mkdir(parents=True)
                    result = {
                        "scene": scene,
                        "arm": arm,
                        "checkpoint_bytes": 1.0,
                        "triangles": 1.0,
                        "vertices": 1.0,
                        "metrics": {
                            "l1": 1.0,
                            "psnr": psnr,
                            "ssim": psnr,
                            "lpips_vgg": 1.0,
                            "fps": 1.0,
                        },
                    }
                    (output / "result.json").write_text(json.dumps(result))
            (formal_root / "formal_table.json").write_text(
                json.dumps({"rows": formal_rows})
            )
            (opacity_root / "ablation_table.json").write_text(
                json.dumps({"rows": opacity_rows})
            )

            with redirect_stdout(io.StringIO()):
                main(formal_root, opacity_root, output_root)

            table = json.loads((output_root / "sensitivity.json").read_text())
            self.assertEqual(table["best_mean"]["psnr"], "ours_opacity08")
            self.assertEqual(table["opacity_floor"]["ours_opacity07"], 0.7)
            self.assertEqual((output_root / "DONE").read_text(), "complete\n")


if __name__ == "__main__":
    unittest.main()
