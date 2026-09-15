"""Three-scene comparison for softmin opacity-gradient routing.

The pooled arm differs from the frozen v1 runs in exactly one respect: how a
face's opacity gradient is split over its three vertices. Everything else -- the
endpoint, the schedules, the renderer, the evaluator -- is identical, so the
per-scene deltas below isolate the routing.

Usage:
    python -m sota.v3_gate <formal_table.json> <runs_root> [scene ...]

Writes ``<runs_root>/gate.json``.
"""

import argparse
import json
from pathlib import Path

ARMS = ("ours_quality", "ours_speed", "ours_opacity")
METRICS = ("psnr", "ssim", "lpips_vgg", "l1", "fps")
# Lower is better for these; the rest improve upward.
LOWER_IS_BETTER = ("lpips_vgg", "l1")
SIZE_KEYS = ("checkpoint_bytes", "triangles", "vertices")


def _mean(values):
    return sum(values) / len(values)


def _load_arm(runs_root, scene, arm):
    path = Path(runs_root) / scene / arm / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = dict(payload["metrics"])
    for key in SIZE_KEYS:
        row[key] = payload[key]
    return row


def _delta(arm, reference, metric):
    """Signed improvement: positive always means the pooled arm is better."""
    change = arm[metric] - reference[metric]
    return -change if metric in LOWER_IS_BETTER else change


def build(formal_table, ablation_table, runs_root, scenes):
    # ours_opacity is a stage of the opacity ablation, not a main-table arm, so
    # each arm is compared against the table that actually froze it.
    reference_rows = {
        "ours_quality": json.loads(Path(formal_table).read_text(encoding="utf-8"))["rows"],
        "ours_speed": json.loads(Path(formal_table).read_text(encoding="utf-8"))["rows"],
        "ours_opacity": json.loads(Path(ablation_table).read_text(encoding="utf-8"))["rows"],
    }
    report = {
        "experiment": "softtail-v3-softmin-opacity-routing",
        "scenes": list(scenes),
        "reference": str(formal_table),
        "per_scene": {},
        "means": {},
        "deltas": {},
        "win_counts": {},
    }

    for arm in ARMS:
        pooled = {scene: _load_arm(runs_root, scene, arm) for scene in scenes}
        baseline = {scene: reference_rows[arm][scene][arm] for scene in scenes}

        report["per_scene"][arm] = {
            scene: {
                "pooled": pooled[scene],
                "v1": baseline[scene],
                "delta": {m: _delta(pooled[scene], baseline[scene], m) for m in METRICS},
            }
            for scene in scenes
        }
        report["means"][arm] = {
            "pooled": {m: _mean([pooled[s][m] for s in scenes]) for m in METRICS},
            "v1": {m: _mean([baseline[s][m] for s in scenes]) for m in METRICS},
        }
        report["deltas"][arm] = {
            m: report["means"][arm]["pooled"][m] - report["means"][arm]["v1"][m]
            for m in METRICS
        }
        report["win_counts"][arm] = {
            m: sum(1 for s in scenes if _delta(pooled[s], baseline[s], m) > 0)
            for m in METRICS
        }
        for key in SIZE_KEYS:
            report["means"][arm]["pooled"][key] = _mean([pooled[s][key] for s in scenes])
            report["means"][arm]["v1"][key] = _mean([baseline[s][key] for s in scenes])

    quality = report["deltas"]["ours_quality"]
    report["headline"] = {
        "psnr_gain_db": quality["psnr"],
        "ssim_gain": quality["ssim"],
        "lpips_gain": -quality["lpips_vgg"],
        "psnr_wins": report["win_counts"]["ours_quality"]["psnr"],
        "scenes": len(scenes),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("formal_table")
    parser.add_argument("ablation_table")
    parser.add_argument("runs_root")
    parser.add_argument("scenes", nargs="*", default=["room", "bicycle", "garden"])
    args = parser.parse_args()

    scenes = args.scenes or ["room", "bicycle", "garden"]
    report = build(args.formal_table, args.ablation_table, args.runs_root, scenes)
    out = Path(args.runs_root) / "gate.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["headline"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
