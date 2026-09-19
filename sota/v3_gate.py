"""Three-scene comparison of a candidate arm against the frozen v1 runs.

The candidate differs from v1 in exactly one declared respect (a gradient
routing, an opacity model); the endpoint, the schedules, the renderer and the
evaluator are identical, so the per-scene deltas below isolate that change.

Usage:
    python -m sota.v3_gate <formal_table.json> <ablation_table.json> <runs_root>
        [scene ...] [--experiment NAME]

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
SCENES = ["room", "bicycle", "garden"]


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
    """Signed improvement: positive always means the candidate is better."""
    change = arm[metric] - reference[metric]
    return -change if metric in LOWER_IS_BETTER else change


def build(formal_table, ablation_table, runs_root, scenes, experiment):
    # ours_opacity is a stage of the opacity ablation, not a main-table arm, so
    # each arm is compared against the table that actually froze it.
    formal_rows = json.loads(Path(formal_table).read_text(encoding="utf-8"))["rows"]
    reference_rows = {
        "ours_quality": formal_rows,
        "ours_speed": formal_rows,
        "ours_opacity": json.loads(Path(ablation_table).read_text(encoding="utf-8"))["rows"],
    }
    report = {
        "experiment": experiment,
        "scenes": list(scenes),
        "reference": str(formal_table),
        "per_scene": {},
        "means": {},
        "deltas": {},
        "win_counts": {},
    }

    for arm in ARMS:
        candidate = {scene: _load_arm(runs_root, scene, arm) for scene in scenes}
        baseline = {scene: reference_rows[arm][scene][arm] for scene in scenes}

        report["per_scene"][arm] = {
            scene: {
                "candidate": candidate[scene],
                "v1": baseline[scene],
                "delta": {m: _delta(candidate[scene], baseline[scene], m) for m in METRICS},
            }
            for scene in scenes
        }
        report["means"][arm] = {
            "candidate": {m: _mean([candidate[s][m] for s in scenes]) for m in METRICS},
            "v1": {m: _mean([baseline[s][m] for s in scenes]) for m in METRICS},
        }
        report["deltas"][arm] = {
            m: report["means"][arm]["candidate"][m] - report["means"][arm]["v1"][m]
            for m in METRICS
        }
        report["win_counts"][arm] = {
            m: sum(1 for s in scenes if _delta(candidate[s], baseline[s], m) > 0)
            for m in METRICS
        }
        for key in SIZE_KEYS:
            report["means"][arm]["candidate"][key] = _mean([candidate[s][key] for s in scenes])
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
    parser.add_argument("scenes", nargs="*")
    parser.add_argument("--experiment", default="softtail-v3-softmin-opacity-routing")
    args = parser.parse_args()

    report = build(args.formal_table, args.ablation_table, args.runs_root,
                   args.scenes or SCENES, args.experiment)
    out = Path(args.runs_root) / "gate.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["headline"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
