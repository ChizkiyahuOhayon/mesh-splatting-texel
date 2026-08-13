"""Seal a complete GoRFE-V1 phase root after all child artifacts exist."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gorfe_v1_io import (
    artifact_records,
    sha256_file,
    source_revision,
    write_json_new,
    write_sha256s,
    write_text_new,
)


REPOSITORY = Path(__file__).resolve().parent
PROTOCOL_PATH = REPOSITORY / "experiments" / "gorfe_v1" / "protocol.md"
CONSTANTS_PATH = REPOSITORY / "experiments" / "gorfe_v1" / "protocol_constants.json"
SCENES = ("garden", "room")
ROOT_OUTPUT_NAMES = frozenset(("manifest.json", "SHA256SUMS", "DONE"))


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_complete(path: Path) -> None:
    if path.read_text(encoding="utf-8") != "complete\n":
        raise RuntimeError(f"malformed completion marker: {path}")


def _require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular artifact is missing: {path}")
    return path


def _validate_prepare(root: Path) -> str:
    for relative in (
        "build.log",
        "garden_preflight.json",
        "gpu.txt",
        "native_extension.so",
        "native_result.json",
        "native_smoke.log",
        "python_env.txt",
        "room_preflight.json",
        "tests.log",
    ):
        _require_file(root, relative)
    wheels = sorted((root / "wheels").glob("diff_triangle_rasterization-*.whl"))
    if len(wheels) != 1 or not wheels[0].is_file() or wheels[0].is_symlink():
        raise RuntimeError("prepare root must contain exactly one regular frozen wheel")
    if _load_json(root / "native_result.json").get("decision") != "pass":
        raise RuntimeError("native exporter gate did not pass")
    for scene in SCENES:
        preflight = _load_json(root / f"{scene}_preflight.json")
        checks = preflight.get("checks")
        if (
            preflight.get("phase") != "prepare-preflight"
            or preflight.get("scene") != scene
            or not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
        ):
            raise RuntimeError(f"{scene} preparation preflight did not pass")
    for scene in SCENES:
        for name in (
            "candidate_manifest.json",
            "candidate_state.pt",
            "result.json",
            "SHA256SUMS",
            "DONE",
        ):
            _require_file(root, f"{scene}/{name}")
        result = _load_json(root / scene / "result.json")
        if result.get("phase") != "prepare" or result.get("decision") != "prepared":
            raise RuntimeError(f"{scene} preparation is not complete")
        _require_complete(root / scene / "DONE")
    return "prepared"


def _validate_evaluate(root: Path) -> str:
    for relative in (
        "extension_verify.json",
        "freeze_verify.json",
        "gpu.txt",
        "install.log",
        "native_result.json",
        "native_smoke.log",
        "python_env.txt",
        "tests.log",
        "decision.json",
        "decision_manifest.json",
        "DECISION_SHA256SUMS",
    ):
        _require_file(root, relative)
    if _load_json(root / "freeze_verify.json").get("decision") != "pass":
        raise RuntimeError("candidate-freeze verification did not pass")
    if _load_json(root / "extension_verify.json").get("decision") != "pass":
        raise RuntimeError("installed native extension differs from the frozen binary")
    if _load_json(root / "native_result.json").get("decision") != "pass":
        raise RuntimeError("frozen native exporter gate did not pass")
    for scene in SCENES:
        for name in (
            "evaluation_state.pt",
            "result.json",
            "manifest.json",
            "SHA256SUMS",
            "DONE",
        ):
            _require_file(root, f"{scene}/{name}")
        result = _load_json(root / scene / "result.json")
        if result.get("phase") != "evaluate" or result.get("scene") != scene:
            raise RuntimeError(f"{scene} evaluation artifact is malformed")
        _require_complete(root / scene / "DONE")
    decision = _load_json(root / "decision.json")
    if decision.get("phase") != "overall" or decision.get("decision") not in (
        "pass",
        "fail",
    ):
        raise RuntimeError("overall evaluation decision is malformed")
    return str(decision["decision"])


def _artifacts_before_seal(root: Path) -> dict[str, Path]:
    paths = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise RuntimeError(f"artifact symlinks are forbidden: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in ROOT_OUTPUT_NAMES:
            continue
        if path.name in ("FAILED", "INVALID"):
            raise RuntimeError(f"cannot seal an invalid attempt containing {relative}")
        paths[relative] = path
    return paths


def _sync_artifacts(paths: dict[str, Path]) -> None:
    for path in paths.values():
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def finalize(root: Path, *, phase: str) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"phase root does not exist: {root}")
    for name in ROOT_OUTPUT_NAMES:
        if (root / name).exists():
            raise FileExistsError(f"refusing to overwrite root artifact: {root / name}")
    if phase == "prepare":
        decision = _validate_prepare(root)
    elif phase == "evaluate":
        decision = _validate_evaluate(root)
    else:
        raise ValueError("phase must be prepare or evaluate")

    paths = _artifacts_before_seal(root)
    _sync_artifacts(paths)
    manifest = {
        "schema": "gorfe-v1-phase-root-v1",
        "experiment": "GoRFE-V1",
        "phase": phase,
        "decision": decision,
        "source_revision": source_revision(),
        "protocol": str(PROTOCOL_PATH.relative_to(REPOSITORY)),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "constants_sha256": sha256_file(CONSTANTS_PATH),
        "artifacts": artifact_records(paths),
    }
    manifest_path = write_json_new(root / "manifest.json", manifest)
    checksum_path = write_sha256s(
        root / "SHA256SUMS", [paths[name] for name in sorted(paths)] + [manifest_path]
    )
    done_path = write_text_new(root / "DONE", "complete\n")
    return {
        "manifest": str(manifest_path),
        "checksums": str(checksum_path),
        "done": str(done_path),
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "evaluate"), required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.root, phase=args.phase), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
