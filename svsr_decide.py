"""Apply the preregistered SVSR-G1 three-scene decision rule."""

import json
from pathlib import Path
from argparse import ArgumentParser


SCENES = ("garden", "room", "stump")


def _load(root):
    root = Path(root)
    with open(root / "results.json", encoding="utf-8") as handle:
        results = json.load(handle)
    with open(root / "svsr_manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return results, manifest


def decide(roots):
    loaded = [_load(root) for root in roots]
    by_scene = {results["scene"]: (results, manifest) for results, manifest in loaded}
    if set(by_scene) != set(SCENES) or len(loaded) != len(SCENES):
        raise ValueError("expected exactly one Garden, Room, and Stump result")

    diagnostics = {}
    garden = by_scene["garden"][0]["summary"]
    garden_psnr_gain = garden["fixed"]["psnr"] - garden["sh"]["psnr"]
    garden_lpips_gain = garden["sh"]["lpips_vgg"] - garden["fixed"]["lpips_vgg"]
    if garden_psnr_gain <= 0 or garden_lpips_gain <= 0:
        raise ValueError("Garden fixed checkpoint is not a valid positive control")
    diagnostics["garden"] = {
        "psnr_retention": (
            garden["footprint"]["psnr"] - garden["sh"]["psnr"]
        ) / garden_psnr_gain,
        "lpips_retention": (
            garden["sh"]["lpips_vgg"] - garden["footprint"]["lpips_vgg"]
        ) / garden_lpips_gain,
    }

    for scene in ("room", "stump"):
        summary = by_scene[scene][0]["summary"]
        regression = summary["sh"]["psnr"] - summary["fixed"]["psnr"]
        if regression <= 0:
            raise ValueError(f"{scene} fixed checkpoint is not a valid negative control")
        diagnostics[scene] = {
            "psnr_recovery": (
                summary["footprint"]["psnr"] - summary["fixed"]["psnr"]
            ) / regression,
            "lpips_no_regression": (
                summary["footprint"]["lpips_vgg"] <= summary["fixed"]["lpips_vgg"]
            ),
        }

    hard_fail = (
        diagnostics["garden"]["psnr_retention"] < 0.5
        or diagnostics["garden"]["lpips_retention"] < 0.5
        or diagnostics["room"]["psnr_recovery"] < 0.25
        or diagnostics["stump"]["psnr_recovery"] < 0.25
    )
    passes = (
        diagnostics["garden"]["psnr_retention"] >= 0.7
        and diagnostics["garden"]["lpips_retention"] >= 0.7
        and diagnostics["room"]["psnr_recovery"] >= 0.5
        and diagnostics["stump"]["psnr_recovery"] >= 0.5
        and diagnostics["room"]["lpips_no_regression"]
        and diagnostics["stump"]["lpips_no_regression"]
    )
    confirmatory = all(manifest["confirmatory_settings"] for _, manifest in by_scene.values())
    verdict = "FAIL" if hard_fail else ("PASS" if passes else "MIXED")
    if not confirmatory:
        verdict = f"EXPLORATORY_{verdict}"
    return {"verdict": verdict, "confirmatory": confirmatory, "diagnostics": diagnostics}


if __name__ == "__main__":
    parser = ArgumentParser(description="Decide the SVSR-G1 gate")
    parser.add_argument("roots", nargs=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    decision = decide(args.roots)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(decision, handle, indent=2)
    print(json.dumps(decision, indent=2))

