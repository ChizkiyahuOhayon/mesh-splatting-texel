"""Create the hash-only payload that must be committed between V1 phases."""

import argparse
import json
from pathlib import Path

from gorfe_v1_io import sha256_file, source_revision, write_json_new
from gorfe_v1_prepare_core import build_candidate_freeze_payload


SCENES = ("garden", "room")
ROOT_ARTIFACTS = (
    "build.log",
    "gpu.txt",
    "native_extension.so",
    "native_result.json",
    "native_smoke.log",
    "python_env.txt",
    "tests.log",
    "manifest.json",
    "SHA256SUMS",
    "DONE",
)
SCENE_ARTIFACTS = (
    "candidate_manifest.json",
    "candidate_state.pt",
    "result.json",
    "DONE",
)


def preparation_artifact_paths(root):
    root = Path(root).resolve()
    artifacts = {name.replace(".", "_"): root / name for name in ROOT_ARTIFACTS}
    wheels = sorted((root / "wheels").glob("diff_triangle_rasterization-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            "preparation must contain exactly one frozen diff_triangle_rasterization wheel"
        )
    artifacts["native_wheel"] = wheels[0]
    for scene in SCENES:
        for name in SCENE_ARTIFACTS:
            logical = f"{scene}_{name.replace('.', '_')}"
            artifacts[logical] = root / scene / name
    return artifacts


def build_payload(root, *, revision):
    root = Path(root).resolve()
    phase_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        phase_manifest.get("schema") != "gorfe-v1-phase-root-v1"
        or phase_manifest.get("phase") != "prepare"
        or phase_manifest.get("decision") != "prepared"
        or phase_manifest.get("source_revision") != revision
    ):
        raise RuntimeError("preparation root manifest is not sealable")
    if (root / "DONE").read_text(encoding="utf-8") != "complete\n":
        raise RuntimeError("preparation root DONE marker is malformed")
    for scene in SCENES:
        result_path = root / scene / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("phase") != "prepare" or result.get("decision") != "prepared":
            raise RuntimeError(f"{scene} preparation is not sealable")
        if (root / scene / "DONE").read_text(encoding="utf-8") != "complete\n":
            raise RuntimeError(f"{scene} preparation DONE marker is malformed")
    native = json.loads((root / "native_result.json").read_text(encoding="utf-8"))
    if native.get("decision") != "pass":
        raise RuntimeError("native exporter gate did not pass")

    repository = Path(__file__).resolve().parent
    protocol = repository / "experiments" / "gorfe_v1" / "protocol.md"
    constants = repository / "experiments" / "gorfe_v1" / "protocol_constants.json"
    return build_candidate_freeze_payload(
        source_revision=revision,
        protocol_sha256=sha256_file(protocol),
        constants_sha256=sha256_file(constants),
        preparation_artifacts=preparation_artifact_paths(root),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.prepare_root, revision=source_revision())
    write_json_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
