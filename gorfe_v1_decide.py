"""Combine the two preregistered GoRFE-V1 scene decisions."""

import argparse
import json
from pathlib import Path

from gorfe_v1 import FAMILY_NAMES, decide_overall, decide_scene_family
from gorfe_v1_io import (
    artifact_records,
    sha256_file,
    source_revision,
    write_json_new,
    write_sha256s,
)


REPOSITORY = Path(__file__).resolve().parent
PROTOCOL_PATH = REPOSITORY / "experiments" / "gorfe_v1" / "protocol.md"
CONSTANTS_PATH = REPOSITORY / "experiments" / "gorfe_v1" / "protocol_constants.json"


def _load_scene(path: Path, expected_scene: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("experiment") != "GoRFE-V1"
        or value.get("phase") != "evaluate"
        or value.get("scene") != expected_scene
    ):
        raise ValueError(f"invalid {expected_scene} evaluation result: {path}")
    families = value.get("families")
    if not isinstance(families, dict) or set(families) != set(FAMILY_NAMES):
        raise ValueError(f"{expected_scene} does not report every locked family")
    for family in FAMILY_NAMES:
        recomputed = decide_scene_family(families[family]["metrics"])
        if families[family].get("decision") != recomputed:
            raise ValueError(f"{expected_scene}/{family} stored decision is inconsistent")
    passing = [family for family in FAMILY_NAMES if families[family]["decision"]["pass"]]
    if value.get("passing_families") != passing:
        raise ValueError(f"{expected_scene} passing-family list is inconsistent")
    expected_decision = "pass" if passing else "fail"
    if value.get("decision") != expected_decision:
        raise ValueError(f"{expected_scene} decision is inconsistent")
    return value


def decide(garden_path: Path, room_path: Path) -> dict:
    paths = {"garden": garden_path.resolve(), "room": room_path.resolve()}
    scenes = {scene: _load_scene(paths[scene], scene) for scene in paths}
    metrics = {
        scene: {
            family: scenes[scene]["families"][family]["metrics"]
            for family in FAMILY_NAMES
        }
        for scene in scenes
    }
    outcome = decide_overall(metrics)
    return {
        "experiment": "GoRFE-V1",
        "phase": "overall",
        **outcome,
        "scene_result_sha256": {
            scene: sha256_file(path) for scene, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--garden-result", type=Path, required=True)
    parser.add_argument("--room-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"evaluation output root does not exist: {output}")
    result = decide(args.garden_result, args.room_result)
    result_path = write_json_new(output / "decision.json", result)
    manifest = {
        "schema": "gorfe-v1-overall-manifest-v1",
        "experiment": "GoRFE-V1",
        "phase": "overall",
        "source_revision": source_revision(),
        "protocol": str(PROTOCOL_PATH.relative_to(REPOSITORY)),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "constants_sha256": sha256_file(CONSTANTS_PATH),
        "inputs": artifact_records(
            {"garden": args.garden_result, "room": args.room_result}
        ),
        "result": artifact_records({"decision": result_path})["decision"],
    }
    manifest_path = write_json_new(output / "decision_manifest.json", manifest)
    write_sha256s(output / "DECISION_SHA256SUMS", [result_path, manifest_path])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
