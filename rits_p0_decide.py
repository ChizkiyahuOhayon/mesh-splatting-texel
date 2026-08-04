"""The locked RITS-P0 decision, separable from the CUDA fitting run."""

import json
from argparse import ArgumentParser
from pathlib import Path


REQUIRED_REDUCTION = 0.80
SCENES = ("garden", "room")


def decide(inherited, projected):
    """Held-out discrepancy checks; the caller adds the integrity checks."""
    return {
        "probe_region_mae_reduced_by_80_percent": (
            inherited["probe_region_mae"] is not None
            and projected["probe_region_mae"] is not None
            and projected["probe_region_mae"]
            <= (1.0 - REQUIRED_REDUCTION) * inherited["probe_region_mae"]
        ),
        "global_mae_improved": projected["global_mae"] < inherited["global_mae"],
    }


def _scene_row(results):
    inherited = results["held_out"]["inherited"]
    projected = results["held_out"]["projected"]
    return {
        "scene_pass": results["decision"]["scene_pass"],
        "inherited_probe_region_mae": inherited["probe_region_mae"],
        "projected_probe_region_mae": projected["probe_region_mae"],
        "reduction": (
            1.0 - projected["probe_region_mae"] / inherited["probe_region_mae"]
            if inherited["probe_region_mae"]
            else None
        ),
        "inherited_global_mae": inherited["global_mae"],
        "projected_global_mae": projected["global_mae"],
        "checks": results["decision"]["checks"],
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="RITS-P0 two-scene decision")
    for scene in SCENES:
        parser.add_argument(f"--{scene}", required=True)
    args = parser.parse_args()
    per_scene = {}
    for scene in SCENES:
        with open(Path(getattr(args, scene)) / "results.json", encoding="utf-8") as handle:
            per_scene[scene] = _scene_row(json.load(handle))
    print(
        json.dumps(
            {
                "pass": all(row["scene_pass"] for row in per_scene.values()),
                "per_scene": per_scene,
            },
            indent=2,
        )
    )
