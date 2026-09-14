"""Three-scene SoftTail v2 gate against the frozen v1 evaluations.

Reads the ``adaptive_*`` arm results under OUTPUT_ROOT/<scene>/<arm>/result.json,
the frozen v1 nine-scene ``formal_table.json`` (stock, ours_speed,
ours_quality) and ``ablation_table.json`` (ours_opacity), and writes
``gate.json`` with the three-scene means, the deltas of the adaptive quality
point against the v1 quality point, and the pass/fail of every gate criterion.
"""

import json
import sys
from pathlib import Path

from sota.formal_table import METRICS, mean
from sota.opacity_sensitivity import extract


SCENES = ("room", "bicycle", "garden")
V1_ARMS = ("stock", "ours_opacity", "ours_speed", "ours_quality")
V2_ARMS = ("adaptive_opacity", "adaptive_speed", "adaptive_quality")
PSNR_GATE_DB = 0.10


def gate_decision(v2, v1):
    """Per-criterion pass flags for the adaptive quality point vs v1 quality."""
    delta = {key: v2[key] - v1[key] for key in METRICS}
    checks = {
        "psnr": delta["psnr"] >= PSNR_GATE_DB,
        "ssim": delta["ssim"] >= 0.0,
        "lpips_vgg": delta["lpips_vgg"] <= 0.0,
        "checkpoint_bytes": delta["checkpoint_bytes"] <= 0.0,
        "triangles": delta["triangles"] <= 0.0,
    }
    return {"delta": delta, "checks": checks, "pass": all(checks.values())}


def load_result(path, scene, arm):
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    if result["scene"] != scene or result["arm"] != arm:
        raise ValueError(f"identity mismatch in {path}")
    return extract(result), result


def main(formal_table, ablation_table, output_root, scenes=SCENES):
    output_root = Path(output_root).resolve()
    with open(formal_table, encoding="utf-8") as handle:
        formal = json.load(handle)
    with open(ablation_table, encoding="utf-8") as handle:
        ablation = json.load(handle)

    rows = {}
    provenance = {}
    for scene in scenes:
        rows[scene] = {
            "stock": formal["rows"][scene]["stock"],
            "ours_speed": formal["rows"][scene]["ours_speed"],
            "ours_quality": formal["rows"][scene]["ours_quality"],
            "ours_opacity": ablation["rows"][scene]["ours_opacity"],
        }
        for arm in V2_ARMS:
            row, result = load_result(output_root / scene / arm / "result.json", scene, arm)
            rows[scene][arm] = row
            provenance[f"{scene}/{arm}"] = {
                key: result.get(key) for key in
                ("source_revision", "checkpoint", "adaptive_opacity", "upsample",
                 "transmittance_threshold", "absorb_transmittance_tail")
            }

    means = {
        arm: {key: mean([rows[scene][arm] for scene in scenes], key) for key in METRICS}
        for arm in V1_ARMS + V2_ARMS
    }
    decision = gate_decision(means["adaptive_quality"], means["ours_quality"])
    per_scene = {
        scene: gate_decision(rows[scene]["adaptive_quality"], rows[scene]["ours_quality"])
        for scene in scenes
    }
    opacity_only_delta = {
        key: means["adaptive_opacity"][key] - means["ours_opacity"][key] for key in METRICS
    }
    table = {
        "experiment": "softtail-v2-three-scene-gate-v1",
        "scenes": list(scenes),
        "psnr_gate_db": PSNR_GATE_DB,
        "rows": rows,
        "means": means,
        "adaptive_quality_vs_v1_quality": decision,
        "adaptive_opacity_vs_v1_opacity_delta": opacity_only_delta,
        "per_scene_adaptive_quality_vs_v1_quality": per_scene,
        "provenance": provenance,
    }
    with open(output_root / "gate.json", "x", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, allow_nan=False)
        handle.write("\n")
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in table.items() if k not in ("rows", "provenance")}, indent=2))
    return table


def compare_control(gate_root, control_root, scenes=("room", "bicycle")):
    """Adaptive vs shuffled-endpoint control at the quality point, same scenes."""
    gate_root = Path(gate_root).resolve()
    control_root = Path(control_root).resolve()
    with open(gate_root / "gate.json", encoding="utf-8") as handle:
        gate = json.load(handle)
    rows = {}
    for scene in scenes:
        control, _ = load_result(control_root / scene / "adaptive_quality" / "result.json",
                                 scene, "adaptive_quality")
        rows[scene] = {
            "adaptive_quality": gate["rows"][scene]["adaptive_quality"],
            "shuffled_quality": control,
            "ours_quality": gate["rows"][scene]["ours_quality"],
        }
    means = {
        arm: {key: mean([rows[scene][arm] for scene in scenes], key) for key in METRICS}
        for arm in ("adaptive_quality", "shuffled_quality", "ours_quality")
    }
    table = {
        "experiment": "softtail-v2-shuffled-endpoint-control-v1",
        "scenes": list(scenes),
        "rows": rows,
        "means": means,
        "adaptive_minus_shuffled": {
            key: means["adaptive_quality"][key] - means["shuffled_quality"][key] for key in METRICS
        },
        "shuffled_minus_v1": {
            key: means["shuffled_quality"][key] - means["ours_quality"][key] for key in METRICS
        },
    }
    with open(control_root / "control.json", "x", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, allow_nan=False)
        handle.write("\n")
    (control_root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in table.items() if k != "rows"}, indent=2))
    return table


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] != "--control":
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "--control":
        compare_control(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(
            "usage: python -m sota.v2_gate FORMAL_TABLE_JSON ABLATION_TABLE_JSON OUTPUT_ROOT\n"
            "       python -m sota.v2_gate --control GATE_ROOT CONTROL_ROOT"
        )
