"""Seal one EdgeVal-E0 representation-gate run."""

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
    override = os.environ.get("EDGEVAL_SOURCE_REVISION")
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
        raise RuntimeError("refusing to seal a non-passing EdgeVal-E0 result")
    import diff_triangle_rasterization._C as extension

    extension_path = Path(extension.__file__).resolve()
    manifest = {
        "experiment": "EdgeVal-E0",
        "protocol": "experiments/edgeval_e0/protocol.md",
        "source_revision": _revision(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "extension_path": str(extension_path),
        "artifacts": {
            "result": {"path": str(args.result), "sha256": _sha256(args.result)},
            "build_log": {"path": str(args.build_log), "sha256": _sha256(args.build_log)},
            "test_log": {"path": str(args.test_log), "sha256": _sha256(args.test_log)},
            "smoke_log": {"path": str(args.smoke_log), "sha256": _sha256(args.smoke_log)},
            "extension": {"path": str(extension_path), "sha256": _sha256(extension_path)},
        },
        "checks": result["checks"],
        "decision": result["decision"],
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
