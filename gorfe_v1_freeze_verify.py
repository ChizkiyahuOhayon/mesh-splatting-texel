"""Verify the tracked GoRFE-V1 candidate freeze before evaluation work."""

import argparse
import json
from pathlib import Path

from gorfe_v1_evaluate import (
    CONSTANTS_PATH,
    PROTOCOL_PATH,
    verify_tracked_freeze_commit,
)
from gorfe_v1_freeze_payload import preparation_artifact_paths
from gorfe_v1_io import sha256_file, write_json_new
from gorfe_v1_prepare_core import verify_candidate_freeze_payload


def verify(prepare_root: Path, freeze_file: Path) -> dict:
    prepare_root = prepare_root.resolve()
    freeze_file = freeze_file.resolve()
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict):
        raise ValueError("candidate-freeze JSON root must be an object")
    implementation_revision = str(freeze.get("source_revision", ""))
    evaluation_revision = verify_tracked_freeze_commit(
        freeze_file, implementation_revision
    )
    verify_candidate_freeze_payload(
        freeze,
        source_revision=implementation_revision,
        protocol_sha256=sha256_file(PROTOCOL_PATH),
        constants_sha256=sha256_file(CONSTANTS_PATH),
        preparation_artifacts=preparation_artifact_paths(prepare_root),
    )
    return {
        "schema": "gorfe-v1-freeze-verification-v1",
        "experiment": "GoRFE-V1",
        "phase": "evaluate-preflight",
        "decision": "pass",
        "implementation_source_revision": implementation_revision,
        "evaluation_revision": evaluation_revision,
        "freeze_file": str(freeze_file),
        "freeze_sha256": sha256_file(freeze_file),
        "prepare_root": str(prepare_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--freeze-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.prepare_root, args.freeze_file)
    write_json_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
