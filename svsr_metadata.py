"""Source provenance for Git checkouts and archive deployments."""

import os
from pathlib import Path
import subprocess


def source_revision():
    override = os.environ.get("SVSR_SOURCE_REVISION")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "No Git metadata found. Archive deployments must set "
            "SVSR_SOURCE_REVISION to the downloaded GitHub revision."
        ) from error


def reserve_output_directory(path):
    path = Path(path)
    try:
        path.mkdir(parents=True)
    except FileExistsError as error:
        raise RuntimeError(f"SVSR output path must not exist: {path}") from error
    return path
