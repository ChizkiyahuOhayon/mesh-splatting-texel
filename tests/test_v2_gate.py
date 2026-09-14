import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sota.formal_table import METRICS
from sota.v2_gate import PSNR_GATE_DB, SCENES, V2_ARMS, compare_control, gate_decision, main


def _row(**overrides):
    row = {key: 1.0 for key in METRICS}
    row.update(overrides)
    return row


def _write_result(root, scene, arm, row):
    output = root / scene / arm
    output.mkdir(parents=True)
    result = {
        "scene": scene,
        "arm": arm,
        "source_revision": "abc",
        "checkpoint": "x.pt",
        "checkpoint_bytes": row["checkpoint_bytes"],
        "triangles": row["triangles"],
        "vertices": row["vertices"],
        "metrics": {key: row[key] for key in ("l1", "psnr", "ssim", "lpips_vgg", "fps")},
    }
    with open(output / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle)


def _write_v1(root):
    formal = {"rows": {scene: {"stock": _row(psnr=24.0), "ours_speed": _row(psnr=24.9),
                               "ours_quality": _row(psnr=25.0, ssim=0.7, lpips_vgg=0.3,
                                                    checkpoint_bytes=100.0, triangles=50.0)}
                       for scene in SCENES}}
    ablation = {"rows": {scene: {"ours_opacity": _row(psnr=24.95)} for scene in SCENES}}
    formal_path = root / "formal_table.json"
    ablation_path = root / "ablation_table.json"
    formal_path.write_text(json.dumps(formal), encoding="utf-8")
    ablation_path.write_text(json.dumps(ablation), encoding="utf-8")
    return formal_path, ablation_path


class GateDecisionTest(unittest.TestCase):
    def test_all_five_criteria_are_required(self):
        v1 = _row(psnr=25.0, ssim=0.7, lpips_vgg=0.3, checkpoint_bytes=100.0, triangles=50.0)
        good = dict(v1, psnr=25.0 + PSNR_GATE_DB, checkpoint_bytes=99.0)
        self.assertTrue(gate_decision(good, v1)["pass"])
        for key, bad_value in (("psnr", 25.09), ("ssim", 0.69), ("lpips_vgg", 0.31),
                               ("checkpoint_bytes", 101.0), ("triangles", 51.0)):
            decision = gate_decision(dict(good, **{key: bad_value}), v1)
            self.assertFalse(decision["pass"], key)
            self.assertFalse(decision["checks"][key], key)


class GateMainTest(unittest.TestCase):
    def test_gate_json_is_written_with_means_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_path, ablation_path = _write_v1(root)
            output_root = root / "softtail_v2_01"
            for scene in SCENES:
                for arm in V2_ARMS:
                    _write_result(output_root, scene, arm,
                                  _row(psnr=25.2, ssim=0.71, lpips_vgg=0.29,
                                       checkpoint_bytes=90.0, triangles=45.0))
            with redirect_stdout(io.StringIO()):
                table = main(formal_path, ablation_path, output_root)
            self.assertTrue((output_root / "gate.json").is_file())
            self.assertTrue((output_root / "DONE").is_file())
            self.assertTrue(table["adaptive_quality_vs_v1_quality"]["pass"])
            self.assertAlmostEqual(table["adaptive_quality_vs_v1_quality"]["delta"]["psnr"], 0.2)
            self.assertAlmostEqual(table["means"]["ours_quality"]["psnr"], 25.0)
            self.assertIn("room/adaptive_quality", table["provenance"])
            for scene in SCENES:
                self.assertTrue(table["per_scene_adaptive_quality_vs_v1_quality"][scene]["pass"])

    def test_identity_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_path, ablation_path = _write_v1(root)
            output_root = root / "out"
            for scene in SCENES:
                for arm in V2_ARMS:
                    _write_result(output_root, scene, arm, _row())
            wrong = output_root / "room" / "adaptive_quality" / "result.json"
            data = json.loads(wrong.read_text(encoding="utf-8"))
            data["scene"] = "garden"
            wrong.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                with redirect_stdout(io.StringIO()):
                    main(formal_path, ablation_path, output_root)


class ControlTest(unittest.TestCase):
    def test_control_compares_adaptive_against_shuffled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_path, ablation_path = _write_v1(root)
            gate_root = root / "gate"
            for scene in SCENES:
                for arm in V2_ARMS:
                    _write_result(gate_root, scene, arm, _row(psnr=25.3))
            with redirect_stdout(io.StringIO()):
                main(formal_path, ablation_path, gate_root)
            control_root = root / "control"
            for scene in ("room", "bicycle"):
                _write_result(control_root, scene, "adaptive_quality", _row(psnr=25.1))
            with redirect_stdout(io.StringIO()):
                table = compare_control(gate_root, control_root)
            self.assertAlmostEqual(table["adaptive_minus_shuffled"]["psnr"], 0.2)
            self.assertAlmostEqual(table["shuffled_minus_v1"]["psnr"], 0.1)
            self.assertTrue((control_root / "control.json").is_file())


if __name__ == "__main__":
    unittest.main()
