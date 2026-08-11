"""Seal one passing GoRFE-V0 synthetic integrity run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

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
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--gate-log", type=Path, required=True)
    parser.add_argument("--python-env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    if result.get("experiment") != "GoRFE-V0" or result.get("decision") != "pass":
        raise RuntimeError("refusing to seal a non-passing GoRFE-V0 result")
    artifacts = {
        "result": args.result,
        "test_log": args.test_log,
        "gate_log": args.gate_log,
        "python_env": args.python_env,
    }
    manifest = {
        "experiment": "GoRFE-V0",
        "protocol": "experiments/gorfe_v0/protocol.md",
        "source_revision": _revision(),
        "fixture": result["fixture"],
        "device": result["device"],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
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
