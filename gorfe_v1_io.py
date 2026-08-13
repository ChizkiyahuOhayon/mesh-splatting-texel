"""Write-once artifact helpers shared by the two GoRFE-V1 phases."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch


def source_revision():
    override = os.environ.get("GORFE_V1_SOURCE_REVISION")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "GoRFE-V1 needs Git metadata or GORFE_V1_SOURCE_REVISION"
        ) from error


def sha256_file(path, *, chunk_bytes=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path):
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)
    return path


def save_torch_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)
    return path


def write_text_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)
    return path


def artifact_records(paths):
    records = {}
    for name, value in sorted(paths.items()):
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        records[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def directory_bytes(path):
    return sum(
        item.stat().st_size
        for item in Path(path).rglob("*")
        if item.is_file()
    )


def write_sha256s(path, artifacts):
    rows = []
    for artifact in artifacts:
        resolved = Path(artifact).resolve()
        rows.append(f"{sha256_file(resolved)}  {resolved}\n")
    return write_text_new(path, "".join(rows))
