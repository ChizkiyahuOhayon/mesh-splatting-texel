"""Aggregate the three preregistered FMMS G0 scene outputs."""

import json
from pathlib import Path
import statistics
from argparse import ArgumentParser


SCENES = ("garden", "room", "stump")
AA_VARIANTS = ("aa1", "aa2")


def quality_pass(summary, variant):
    delta = summary[variant]["delta_vs_ssaa4"]
    return (
        delta["psnr"] >= -0.10
        and delta["ssim"] >= -0.003
        and delta["lpips_vgg"] <= 0.005
    )


def decide(run_dirs):
    records = {}
    for run_dir in run_dirs:
        root = Path(run_dir)
        with open(root / "results.json", encoding="utf-8") as handle:
            results = json.load(handle)
        with open(root / "timing.json", encoding="utf-8") as handle:
            timing = json.load(handle)
        if results["scene"] != timing["scene"]:
            raise ValueError(f"Scene mismatch in {root}")
        records[results["scene"]] = {"results": results, "timing": timing}

    if set(records) != set(SCENES):
        raise ValueError(f"Expected exactly {SCENES}, got {tuple(sorted(records))}")

    verdicts = {}
    for variant in AA_VARIANTS:
        scene_quality = {
            scene: quality_pass(records[scene]["results"]["summary"], variant)
            for scene in SCENES
        }
        ssaa_times = [
            sample
            for scene in SCENES
            for sample in records[scene]["timing"]["variants"]["ssaa4"]["samples_ms"]
        ]
        native_times = [
            sample
            for scene in SCENES
            for sample in records[scene]["timing"]["variants"][variant]["samples_ms"]
        ]
        speedup = statistics.median(ssaa_times) / statistics.median(native_times)
        ssaa_memory = statistics.median(
            records[scene]["timing"]["variants"]["ssaa4"]["peak_increment_bytes"]
            for scene in SCENES
        )
        native_memory = statistics.median(
            records[scene]["timing"]["variants"][variant]["peak_increment_bytes"]
            for scene in SCENES
        )
        memory_reduction = (
            ssaa_memory / native_memory
            if ssaa_memory > 0 and native_memory > 0
            else 0.0
        )
        verdicts[variant] = {
            "quality_by_scene": scene_quality,
            "speedup": speedup,
            "memory_reduction": memory_reduction,
            "passes": all(scene_quality.values()) and max(speedup, memory_reduction) >= 4.0,
        }

    winner = next((variant for variant in AA_VARIANTS if verdicts[variant]["passes"]), None)
    return {"verdict": "PASS" if winner else "FAIL", "winner": winner, "variants": verdicts}


if __name__ == "__main__":
    parser = ArgumentParser(description="Apply the preregistered FMMS G0 decision rule")
    parser.add_argument("run_dirs", nargs=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decision = decide(args.run_dirs)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    print(json.dumps(decision, indent=2))
