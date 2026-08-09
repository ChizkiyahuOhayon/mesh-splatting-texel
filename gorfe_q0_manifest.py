"""Seal one GoRFE-Q0 representation-gate run."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision():
    override = os.environ.get("GORFE_SOURCE_REVISION")
    if override:
        return override
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--smoke-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    if result.get("decision") != "pass":
        raise RuntimeError("refusing to seal a non-passing GoRFE-Q0 result")
    import diff_triangle_rasterization._C as extension

    extension_path = Path(extension.__file__).resolve()
    artifacts = {
        "result": args.result,
        "build_log": args.build_log,
        "test_log": args.test_log,
        "smoke_log": args.smoke_log,
        "extension": extension_path,
    }
    manifest = {
        "experiment": "GoRFE-Q0",
        "protocol": "experiments/gorfe_q0/protocol.md",
        "source_revision": _revision(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "extension_path": str(extension_path),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
        "checks": result["checks"],
        "decision": result["decision"],
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
