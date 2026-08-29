"""Dependency-free helpers for DTU mesh culling."""

from pathlib import Path


def select_dtu_masks(mask_dir, count):
    """Return the official zero-padded DTU masks in camera order."""
    mask_dir = Path(mask_dir)
    paths = [mask_dir / f"{index:03d}.png" for index in range(count)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(
            f"missing {len(missing)} DTU masks in {mask_dir}: "
            f"{', '.join(missing[:5])}"
        )
    return paths
