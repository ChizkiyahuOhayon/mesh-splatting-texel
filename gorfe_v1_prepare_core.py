"""Target-free preparation primitives for the preregistered GoRFE-V1 gate.

This module intentionally has no image or renderer dependency.  It reads COLMAP
camera *metadata*, validates the frozen MeshSplatting checkpoint, constructs the
canonical candidate topology, and provides the identities used by the separate
candidate-freeze commit.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import struct
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


UINT64_MASK = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_MUL1 = 0xBF58476D1CE4E5B9
SPLITMIX_MUL2 = 0x94D049BB133111EB
OFFICIAL_HOLD = 8

_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
_CAMERA_MODE_NAMES = {name: (model_id, count) for model_id, (name, count) in _CAMERA_MODELS.items()}
_TARGET_SUFFIXES = frozenset(
    {".bmp", ".exr", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


class TargetAccessError(RuntimeError):
    """Raised when a preparation process attempts to read a target image."""


class IdentityDriftError(RuntimeError):
    """Raised when an input no longer matches its sealed identity."""


@dataclass(frozen=True)
class MetadataCamera:
    image_id: int
    camera_id: int
    image_name: str
    relative_image_path: str
    width: int
    height: int
    model: str
    parameters: tuple[float, ...]
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    fold: int


@dataclass(frozen=True)
class ColmapMetadataSplit:
    train_cameras: tuple[MetadataCamera, ...]
    test_names: tuple[str, ...]
    train_name_sha256: str
    test_name_sha256: str
    fold_map_sha256: str
    fold_sizes: tuple[int, ...]
    metadata_format: str

    @property
    def fold_map(self) -> dict[str, int]:
        return {camera.image_name: camera.fold for camera in self.train_cameras}


@dataclass(frozen=True)
class CheckpointInspection:
    path: str
    sha256: str
    bytes: int
    keys: tuple[str, ...]
    tensor_shapes: dict[str, list[int]]
    tensor_dtypes: dict[str, str]
    vertex_count: int
    face_count: int
    active_sh_degree: int
    texel_order: int
    activated_sigma: float

    def to_manifest(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "keys": list(self.keys),
            "tensor_shapes": self.tensor_shapes,
            "tensor_dtypes": self.tensor_dtypes,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "active_sh_degree": self.active_sh_degree,
            "texel_order": self.texel_order,
            "activated_sigma": self.activated_sigma,
        }


@dataclass(frozen=True)
class CandidateTopology:
    edge_count: int
    vertex_stride: int
    candidate_edge_indices: torch.Tensor
    candidate_edges: torch.Tensor
    candidate_hashes: np.ndarray
    candidate_incident_face_counts: torch.Tensor
    face_candidate_edges: torch.Tensor


def sha256_file(path: os.PathLike[str] | str, *, chunk_bytes: int = 1 << 20) -> str:
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def name_list_sha256(names: Sequence[str]) -> str:
    """Hash sorted immutable names using the exact preregistered byte convention."""
    ordered = _ordered_unique_names(names)
    digest = hashlib.sha256()
    for name in ordered:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def deterministic_fold_map(names: Sequence[str], *, fold_count: int = 4) -> dict[str, int]:
    if not isinstance(fold_count, int) or fold_count < 2:
        raise ValueError("fold_count must be an integer of at least two")
    ordered = _ordered_unique_names(names)
    return {name: rank % fold_count for rank, name in enumerate(ordered)}


def fold_map_sha256(fold_map: Mapping[str, int]) -> str:
    checked: dict[str, int] = {}
    for name, fold in fold_map.items():
        _validate_name(name)
        if not isinstance(fold, int) or isinstance(fold, bool) or fold < 0:
            raise ValueError(f"invalid fold for camera {name!r}: {fold!r}")
        checked[name] = fold
    if len(checked) != len(fold_map):
        raise ValueError("fold-map camera identities must be unique")
    return canonical_json_sha256(checked)


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("camera names must be nonempty strings")
    if "\n" in name or "\r" in name or "\x00" in name:
        raise ValueError(f"camera name contains a forbidden delimiter: {name!r}")


def _ordered_unique_names(names: Sequence[str]) -> list[str]:
    checked = list(names)
    for name in checked:
        _validate_name(name)
    if len(checked) != len(set(checked)):
        raise ValueError("camera image_name values must be unique")
    return sorted(checked, key=lambda name: name.encode("utf-8"))


def _read_exact(handle, byte_count: int, context: str) -> bytes:
    data = handle.read(byte_count)
    if len(data) != byte_count:
        raise ValueError(f"truncated COLMAP {context}")
    return data


def _read_cameras_binary(path: Path) -> dict[int, tuple[str, int, int, tuple[float, ...]]]:
    cameras = {}
    with path.open("rb") as handle:
        count = struct.unpack("<Q", _read_exact(handle, 8, "camera count"))[0]
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ", _read_exact(handle, 24, "camera header")
            )
            if model_id not in _CAMERA_MODELS:
                raise ValueError(f"unknown COLMAP camera model id {model_id}")
            model, parameter_count = _CAMERA_MODELS[model_id]
            parameters = struct.unpack(
                "<" + "d" * parameter_count,
                _read_exact(handle, 8 * parameter_count, "camera parameters"),
            )
            if camera_id in cameras:
                raise ValueError(f"duplicate COLMAP camera id {camera_id}")
            cameras[camera_id] = (model, int(width), int(height), tuple(parameters))
    return cameras


def _read_cameras_text(path: Path) -> dict[int, tuple[str, int, int, tuple[float, ...]]]:
    cameras = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError("malformed COLMAP cameras.txt row")
            camera_id = int(fields[0])
            model = fields[1]
            if model not in _CAMERA_MODE_NAMES:
                raise ValueError(f"unknown COLMAP camera model {model!r}")
            parameter_count = _CAMERA_MODE_NAMES[model][1]
            if len(fields) != 4 + parameter_count:
                raise ValueError(f"camera {camera_id} has the wrong parameter count")
            if camera_id in cameras:
                raise ValueError(f"duplicate COLMAP camera id {camera_id}")
            cameras[camera_id] = (
                model,
                int(fields[2]),
                int(fields[3]),
                tuple(float(value) for value in fields[4:]),
            )
    return cameras


def _read_null_terminated_utf8(handle, *, limit: int = 1 << 20) -> str:
    value = bytearray()
    while len(value) <= limit:
        byte = _read_exact(handle, 1, "image name")
        if byte == b"\x00":
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("COLMAP image name is not valid UTF-8") from error
        value.extend(byte)
    raise ValueError("COLMAP image name exceeds the safety limit")


def _read_images_binary(path: Path) -> list[tuple[int, tuple[float, ...], tuple[float, ...], int, str]]:
    images = []
    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        count = struct.unpack("<Q", _read_exact(handle, 8, "image count"))[0]
        seen_ids = set()
        for _ in range(count):
            values = struct.unpack("<idddddddi", _read_exact(handle, 64, "image header"))
            image_id = int(values[0])
            if image_id in seen_ids:
                raise ValueError(f"duplicate COLMAP image id {image_id}")
            seen_ids.add(image_id)
            name = _read_null_terminated_utf8(handle)
            point_count = struct.unpack("<Q", _read_exact(handle, 8, "point count"))[0]
            skip_bytes = 24 * point_count
            if skip_bytes > file_bytes - handle.tell():
                raise ValueError("truncated COLMAP point observations")
            handle.seek(skip_bytes, os.SEEK_CUR)
            images.append((image_id, tuple(values[1:5]), tuple(values[5:8]), int(values[8]), name))
    return images


def _read_images_text(path: Path) -> list[tuple[int, tuple[float, ...], tuple[float, ...], int, str]]:
    images = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError("malformed COLMAP images.txt image row")
            image_id = int(fields[0])
            if image_id in seen_ids:
                raise ValueError(f"duplicate COLMAP image id {image_id}")
            seen_ids.add(image_id)
            qvec = tuple(float(value) for value in fields[1:5])
            tvec = tuple(float(value) for value in fields[5:8])
            camera_id = int(fields[8])
            name = fields[9]
            if handle.readline() == "":
                raise ValueError("COLMAP images.txt is missing a points row")
            images.append((image_id, qvec, tvec, camera_id, name))
    return images


def _image_name(relative_path: str) -> str:
    # Match the published loader exactly: basename followed by the prefix before
    # the first dot, rather than pathlib's last-suffix stem.
    name = os.path.basename(relative_path).split(".")[0]
    _validate_name(name)
    return name


def read_colmap_metadata(
    dataset_root: os.PathLike[str] | str,
    *,
    fold_count: int = 4,
    official_hold: int = OFFICIAL_HOLD,
) -> ColmapMetadataSplit:
    """Read the official split without opening or decoding any image file.

    Test-camera pose and intrinsic records are discarded.  Only their immutable
    names survive, which makes accidental test-camera rendering impossible from
    the returned object.
    """
    if official_hold != OFFICIAL_HOLD:
        raise ValueError(f"GoRFE-V1 requires the official hold value {OFFICIAL_HOLD}")
    sparse = Path(dataset_root).resolve() / "sparse" / "0"
    binary = (sparse / "images.bin", sparse / "cameras.bin")
    text = (sparse / "images.txt", sparse / "cameras.txt")
    if all(path.is_file() for path in binary):
        images = _read_images_binary(binary[0])
        cameras = _read_cameras_binary(binary[1])
        metadata_format = "colmap_binary"
    elif all(path.is_file() for path in text):
        images = _read_images_text(text[0])
        cameras = _read_cameras_text(text[1])
        metadata_format = "colmap_text"
    else:
        raise FileNotFoundError(f"complete COLMAP images/cameras metadata not found under {sparse}")

    named = []
    for image_id, qvec, tvec, camera_id, relative_path in images:
        if camera_id not in cameras:
            raise ValueError(f"COLMAP image {image_id} references unknown camera {camera_id}")
        named.append((_image_name(relative_path), image_id, qvec, tvec, camera_id, relative_path))
    names = [row[0] for row in named]
    ordered_names = _ordered_unique_names(names)
    by_name = {row[0]: row for row in named}
    test_names = tuple(ordered_names[::official_hold])
    train_names = [name for rank, name in enumerate(ordered_names) if rank % official_hold]
    folds = deterministic_fold_map(train_names, fold_count=fold_count)

    train_cameras = []
    for name in train_names:
        _, image_id, qvec, tvec, camera_id, relative_path = by_name[name]
        model, width, height, parameters = cameras[camera_id]
        if model not in ("SIMPLE_PINHOLE", "PINHOLE"):
            raise ValueError(
                f"GoRFE-V1 supports only undistorted SIMPLE_PINHOLE/PINHOLE cameras, got {model}"
            )
        if width < 1 or height < 1 or not all(math.isfinite(value) for value in parameters):
            raise ValueError(f"camera {camera_id} has invalid intrinsic metadata")
        if not all(math.isfinite(value) for value in (*qvec, *tvec)):
            raise ValueError(f"image {image_id} has nonfinite extrinsic metadata")
        train_cameras.append(
            MetadataCamera(
                image_id=image_id,
                camera_id=camera_id,
                image_name=name,
                relative_image_path=relative_path,
                width=width,
                height=height,
                model=model,
                parameters=parameters,
                qvec=qvec,  # type: ignore[arg-type]
                tvec=tvec,  # type: ignore[arg-type]
                fold=folds[name],
            )
        )

    fold_sizes = tuple(sum(camera.fold == fold for camera in train_cameras) for fold in range(fold_count))
    return ColmapMetadataSplit(
        train_cameras=tuple(train_cameras),
        test_names=test_names,
        train_name_sha256=name_list_sha256(train_names),
        test_name_sha256=name_list_sha256(test_names),
        fold_map_sha256=fold_map_sha256(folds),
        fold_sizes=fold_sizes,
        metadata_format=metadata_format,
    )


def validate_scene_camera_identity(
    scene: str, split: ColmapMetadataSplit, constants: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        counts = constants["dataset_view_counts"][scene]
        hashes = constants["dataset_name_hashes"][scene]
        fold_count = int(constants["camera_folds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"protocol constants do not define scene {scene!r}") from error
    observed = {
        "train_count": len(split.train_cameras),
        "test_count": len(split.test_names),
        "train_name_sha256": split.train_name_sha256,
        "test_name_sha256": split.test_name_sha256,
        "fold_map_sha256": split.fold_map_sha256,
        "fold_sizes": list(split.fold_sizes),
    }
    expected_fold_sizes = [
        sum(rank % fold_count == fold for rank in range(int(counts["train"])))
        for fold in range(fold_count)
    ]
    checks = {
        "train_count": observed["train_count"] == int(counts["train"]),
        "test_count": observed["test_count"] == int(counts["test"]),
        "train_name_sha256": observed["train_name_sha256"] == hashes["train"],
        "test_name_sha256": observed["test_name_sha256"] == hashes["test"],
        "fold_sizes": observed["fold_sizes"] == expected_fold_sizes,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise IdentityDriftError(f"{scene} camera metadata drift: {', '.join(failed)}")
    return {"scene": scene, "observed": observed, "checks": checks}


def resolve_checkpoint_path(
    model_root: os.PathLike[str] | str, *, iteration: int = 30000
) -> Path:
    if iteration != 30000:
        raise ValueError("GoRFE-V1 requires the explicit iteration 30000 checkpoint")
    root = Path(model_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model root is not a directory: {root}")
    checkpoint = root / "point_cloud" / "iteration_30000" / "point_cloud_state_dict.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"explicit iteration-30000 checkpoint not found: {checkpoint}")
    return checkpoint


def _scalar(value: Any, name: str) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"checkpoint {name} must be scalar")
        value = value.detach().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"checkpoint {name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"checkpoint {name} must be finite")
    return result


def _integer_scalar(value: Any, name: str) -> int:
    if isinstance(value, bool) or (torch.is_tensor(value) and value.dtype == torch.bool):
        raise TypeError(f"checkpoint {name} must be an integer, not bool")
    numeric = _scalar(value, name)
    if not numeric.is_integer():
        raise ValueError(f"checkpoint {name} must be an integer")
    return int(numeric)


def _require_tensor(state: Mapping[str, Any], name: str, shape: tuple[int | None, ...]) -> torch.Tensor:
    value = state[name]
    if not torch.is_tensor(value):
        raise TypeError(f"checkpoint {name} must be a tensor")
    if value.ndim != len(shape) or any(
        expected is not None and observed != expected
        for observed, expected in zip(value.shape, shape)
    ):
        raise ValueError(f"checkpoint {name} has invalid shape {tuple(value.shape)}")
    return value


def validate_checkpoint_state(
    state: Mapping[str, Any],
    *,
    expected_sh_degree: int = 3,
    expected_texel_order: int = 0,
    expected_sigma: float = 1e-4,
    sigma_abs_tolerance: float = 1e-12,
) -> dict[str, Any]:
    if any(not isinstance(key, str) for key in state):
        raise TypeError("checkpoint keys must be strings")
    if expected_sigma <= 0 or not math.isfinite(expected_sigma):
        raise ValueError("expected_sigma must be finite and positive")
    if sigma_abs_tolerance < 0 or not math.isfinite(sigma_abs_tolerance):
        raise ValueError("sigma_abs_tolerance must be finite and nonnegative")
    required = {
        "_triangle_indices",
        "active_sh_degree",
        "features_dc",
        "features_rest",
        "image_size",
        "importance_score",
        "pixel_count",
        "sigma",
        "texel_order",
        "triangles_points",
        "vertex_weight",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {', '.join(missing)}")

    vertices = _require_tensor(state, "triangles_points", (None, 3))
    faces = _require_tensor(state, "_triangle_indices", (None, 3))
    vertex_count, face_count = int(vertices.shape[0]), int(faces.shape[0])
    vertex_weight = _require_tensor(state, "vertex_weight", (vertex_count, 1))
    features_dc = _require_tensor(state, "features_dc", (vertex_count, 1, 3))
    features_rest = _require_tensor(state, "features_rest", (vertex_count, 15, 3))
    importance_score = _require_tensor(state, "importance_score", (face_count,))
    image_size = _require_tensor(state, "image_size", (face_count,))
    pixel_count = _require_tensor(state, "pixel_count", (face_count,))
    if vertex_count < 1 or face_count < 1:
        raise ValueError("checkpoint mesh must contain vertices and faces")
    if faces.dtype not in (torch.int32, torch.int64):
        raise TypeError("checkpoint _triangle_indices must use int32 or int64")
    if not vertices.is_floating_point():
        raise TypeError("checkpoint triangles_points must be floating point")
    for name, tensor in (
        ("vertex_weight", vertex_weight),
        ("features_dc", features_dc),
        ("features_rest", features_rest),
        ("importance_score", importance_score),
        ("image_size", image_size),
    ):
        if not tensor.is_floating_point():
            raise TypeError(f"checkpoint {name} must be floating point")
    if pixel_count.dtype not in (torch.int32, torch.int64):
        raise TypeError("checkpoint pixel_count must use int32 or int64")
    if faces.numel() and (bool((faces < 0).any()) or int(faces.max()) >= vertex_count):
        raise ValueError("checkpoint faces contain an out-of-range vertex index")
    for key, value in state.items():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"checkpoint tensor {key} contains a nonfinite value")

    active_sh_degree = _integer_scalar(state["active_sh_degree"], "active_sh_degree")
    texel_order = _integer_scalar(state["texel_order"], "texel_order")
    try:
        activated_sigma = math.exp(_scalar(state["sigma"], "sigma"))
    except OverflowError as error:
        raise ValueError("checkpoint activated sigma is not finite") from error
    if active_sh_degree != expected_sh_degree:
        raise ValueError(
            f"checkpoint active_sh_degree is {active_sh_degree}, expected {expected_sh_degree}"
        )
    if texel_order != expected_texel_order:
        raise ValueError(f"checkpoint texel_order is {texel_order}, expected {expected_texel_order}")
    if "texels" in state:
        raise ValueError("a texel_order=0 GoRFE-V1 checkpoint must not contain texels")
    if not math.isclose(activated_sigma, expected_sigma, rel_tol=0.0, abs_tol=sigma_abs_tolerance):
        raise ValueError(
            f"checkpoint activated sigma is {activated_sigma:.17g}, expected {expected_sigma:.17g}"
        )
    return {
        "keys": sorted(str(key) for key in state),
        "tensor_shapes": {
            str(key): list(value.shape) for key, value in sorted(state.items()) if torch.is_tensor(value)
        },
        "tensor_dtypes": {
            str(key): str(value.dtype) for key, value in sorted(state.items()) if torch.is_tensor(value)
        },
        "vertex_count": vertex_count,
        "face_count": face_count,
        "active_sh_degree": active_sh_degree,
        "texel_order": texel_order,
        "activated_sigma": activated_sigma,
    }


def load_validated_checkpoint(
    model_root: os.PathLike[str] | str,
    *,
    iteration: int = 30000,
    expected_sha256: str | None = None,
    expected_sh_degree: int = 3,
    expected_texel_order: int = 0,
    expected_sigma: float = 1e-4,
    sigma_abs_tolerance: float = 1e-12,
) -> tuple[Mapping[str, Any], CheckpointInspection]:
    checkpoint = resolve_checkpoint_path(model_root, iteration=iteration)
    digest_before = sha256_file(checkpoint)
    if expected_sha256 is not None and digest_before != expected_sha256:
        raise IdentityDriftError("checkpoint SHA-256 does not match the frozen identity")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint root object must be a mapping")
    digest_after = sha256_file(checkpoint)
    if digest_after != digest_before:
        raise IdentityDriftError("checkpoint changed while it was being inspected")
    details = validate_checkpoint_state(
        state,
        expected_sh_degree=expected_sh_degree,
        expected_texel_order=expected_texel_order,
        expected_sigma=expected_sigma,
        sigma_abs_tolerance=sigma_abs_tolerance,
    )
    inspection = CheckpointInspection(
        path=str(checkpoint),
        sha256=digest_before,
        bytes=checkpoint.stat().st_size,
        keys=tuple(details["keys"]),
        tensor_shapes=details["tensor_shapes"],
        tensor_dtypes=details["tensor_dtypes"],
        vertex_count=details["vertex_count"],
        face_count=details["face_count"],
        active_sh_degree=details["active_sh_degree"],
        texel_order=details["texel_order"],
        activated_sigma=details["activated_sigma"],
    )
    return state, inspection


def splitmix64_priority(index: int, seed: int) -> int:
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("index must be a nonnegative integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= UINT64_MASK:
        raise ValueError("seed must be an unsigned 64-bit integer")
    z = (index + seed + SPLITMIX_GAMMA) & UINT64_MASK
    z = ((z ^ (z >> 30)) * SPLITMIX_MUL1) & UINT64_MASK
    z = ((z ^ (z >> 27)) * SPLITMIX_MUL2) & UINT64_MASK
    return (z ^ (z >> 31)) & UINT64_MASK


def _splitmix64_vector(indices: np.ndarray, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = indices.astype(np.uint64, copy=False) + np.uint64(seed) + np.uint64(SPLITMIX_GAMMA)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(SPLITMIX_MUL1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(SPLITMIX_MUL2)
        return z ^ (z >> np.uint64(31))


def select_splitmix64_indices(
    edge_count: int,
    *,
    seed: int,
    cap: int = 131072,
    chunk_size: int = 1 << 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the smallest unsigned priorities with bounded temporary memory.

    Returned indices are in canonical-edge order; hashes are aligned with them.
    """
    if not isinstance(edge_count, int) or isinstance(edge_count, bool) or edge_count < 0:
        raise ValueError("edge_count must be a nonnegative integer")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        raise ValueError("cap must be a nonnegative integer")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    splitmix64_priority(0, seed)
    keep = min(edge_count, cap)
    if keep == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint64)

    best_indices = np.empty(0, dtype=np.int64)
    best_hashes = np.empty(0, dtype=np.uint64)
    for start in range(0, edge_count, chunk_size):
        stop = min(start + chunk_size, edge_count)
        indices = np.arange(start, stop, dtype=np.int64)
        hashes = _splitmix64_vector(indices.astype(np.uint64), seed)
        local_keep = min(keep, indices.size)
        if local_keep < indices.size:
            take = np.argpartition(hashes, local_keep - 1)[:local_keep]
            indices, hashes = indices[take], hashes[take]
        indices = np.concatenate((best_indices, indices))
        hashes = np.concatenate((best_hashes, hashes))
        if indices.size > keep:
            take = np.argpartition(hashes, keep - 1)[:keep]
            indices, hashes = indices[take], hashes[take]
        best_indices, best_hashes = indices, hashes

    # SplitMix64 is bijective.  Canonical index is the endpoint-order tie break
    # prescribed defensively by the protocol.
    priority_order = np.lexsort((best_indices, best_hashes))
    best_indices, best_hashes = best_indices[priority_order], best_hashes[priority_order]
    canonical_order = np.argsort(best_indices, kind="stable")
    return best_indices[canonical_order], best_hashes[canonical_order]


def _validate_faces(faces: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not torch.is_tensor(faces) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be a [F, 3] tensor")
    if faces.dtype not in (torch.int32, torch.int64):
        raise TypeError("faces must use int32 or int64 indices")
    if faces.device.type != "cpu":
        raise ValueError("target-free topology construction requires CPU faces")
    if faces.numel() == 0:
        return faces, 0
    if bool((faces < 0).any()):
        raise ValueError("faces contain a negative vertex index")
    degenerate = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    if bool(degenerate.any()):
        row = int(torch.nonzero(degenerate, as_tuple=False)[0, 0])
        raise ValueError(f"face {row} repeats a vertex")
    return faces, int(faces.max()) + 1


def _packed_face_edge_keys(
    faces: torch.Tensor, stride: int, *, chunk_faces: int
) -> torch.Tensor:
    rows = torch.empty((faces.shape[0], 3), dtype=torch.int64)
    for start in range(0, faces.shape[0], chunk_faces):
        stop = min(start + chunk_faces, faces.shape[0])
        local = faces[start:stop].to(torch.int64)
        a, b, c = local.unbind(dim=1)
        rows[start:stop, 0] = torch.minimum(a, b) * stride + torch.maximum(a, b)
        rows[start:stop, 1] = torch.minimum(b, c) * stride + torch.maximum(b, c)
        rows[start:stop, 2] = torch.minimum(c, a) * stride + torch.maximum(c, a)
    return rows


def build_candidate_topology(
    faces: torch.Tensor,
    *,
    seed: int,
    cap: int = 131072,
    chunk_faces: int = 1 << 20,
    hash_chunk_size: int = 1 << 20,
) -> CandidateTopology:
    """Build canonical edges and only the compact candidate face-edge map.

    The global 3F-to-E inverse is deliberately never materialized.  Once the
    canonical unique keys have selected the candidate subset, face rows are
    replayed in chunks and mapped directly to compact candidate IDs.
    """
    if not isinstance(chunk_faces, int) or chunk_faces < 1:
        raise ValueError("chunk_faces must be positive")
    faces, stride = _validate_faces(faces)
    face_count = int(faces.shape[0])
    if face_count == 0:
        empty_long = torch.empty((0,), dtype=torch.int64)
        return CandidateTopology(
            edge_count=0,
            vertex_stride=0,
            candidate_edge_indices=empty_long,
            candidate_edges=torch.empty((0, 2), dtype=torch.int64),
            candidate_hashes=np.empty(0, dtype=np.uint64),
            candidate_incident_face_counts=torch.empty((0,), dtype=torch.uint8),
            face_candidate_edges=torch.empty((0, 3), dtype=torch.int32),
        )
    if stride > torch.iinfo(torch.int64).max // stride:
        raise OverflowError("vertex index range is too large for packed edge keys")

    face_keys = _packed_face_edge_keys(faces, stride, chunk_faces=chunk_faces)
    unique_keys, incident_counts = torch.unique(
        face_keys.reshape(-1), sorted=True, return_counts=True
    )
    nonmanifold = incident_counts > 2
    if bool(nonmanifold.any()):
        row = int(torch.nonzero(nonmanifold, as_tuple=False)[0, 0])
        key = int(unique_keys[row])
        raise ValueError(
            f"non-manifold edge ({key // stride}, {key % stride}) has "
            f"{int(incident_counts[row])} incident faces"
        )
    edge_count = int(unique_keys.numel())
    selected_np, hashes = select_splitmix64_indices(
        edge_count, seed=seed, cap=cap, chunk_size=hash_chunk_size
    )
    selected = torch.from_numpy(selected_np.copy()).to(torch.int64)
    candidate_keys = unique_keys[selected]
    candidate_counts = incident_counts[selected].to(torch.uint8)
    candidate_edges = torch.stack((candidate_keys // stride, candidate_keys % stride), dim=1)
    del unique_keys, incident_counts

    compact = torch.full((face_count, 3), -1, dtype=torch.int32)
    candidate_count = int(candidate_keys.numel())
    if candidate_count:
        for start in range(0, face_count, chunk_faces):
            stop = min(start + chunk_faces, face_count)
            keys = face_keys[start:stop].reshape(-1)
            locations = torch.searchsorted(candidate_keys, keys)
            in_range = locations < candidate_count
            safe = locations.clamp(max=candidate_count - 1)
            matched = in_range & (candidate_keys[safe] == keys)
            mapped = torch.full_like(locations, -1, dtype=torch.int32)
            mapped[matched] = locations[matched].to(torch.int32)
            compact[start:stop] = mapped.reshape(-1, 3)

    return CandidateTopology(
        edge_count=edge_count,
        vertex_stride=stride,
        candidate_edge_indices=selected,
        candidate_edges=candidate_edges,
        candidate_hashes=hashes,
        candidate_incident_face_counts=candidate_counts,
        face_candidate_edges=compact,
    )


class OfficialSplitGuard:
    """Allow only immutable official-training identities after metadata split."""

    def __init__(self, split: ColmapMetadataSplit):
        self._train = frozenset(camera.image_name for camera in split.train_cameras)
        self._test = frozenset(split.test_names)

    def require_training(self, image_name: str) -> None:
        if image_name in self._test:
            raise TargetAccessError(f"official test camera access is forbidden: {image_name}")
        if image_name not in self._train:
            raise TargetAccessError(f"unknown camera identity: {image_name}")


class TargetDecodeSentinel:
    """Context manager that makes any target-image read a sticky failure.

    Paths outside the explicitly supplied image roots, including COLMAP
    ``images.bin`` metadata, remain readable.  A blocked attempt remains recorded
    even if downstream code catches :class:`TargetAccessError`.
    """

    def __init__(self, image_roots: Sequence[os.PathLike[str] | str]):
        roots = tuple(Path(root).expanduser().resolve() for root in image_roots)
        if not roots:
            raise ValueError("at least one target-image root is required")
        self._roots = roots
        self._attempts: list[str] = []
        self._patches: list[tuple[Any, str, Any]] = []

    @property
    def attempted_paths(self) -> tuple[str, ...]:
        return tuple(self._attempts)

    def _is_target(self, value: Any) -> bool:
        if isinstance(value, int) or not isinstance(value, (str, bytes, os.PathLike)):
            return False
        try:
            path = Path(os.fsdecode(value)).expanduser().resolve()
        except (OSError, TypeError, ValueError):
            return False
        if path.suffix.lower() not in _TARGET_SUFFIXES:
            return False
        return any(path == root or root in path.parents for root in self._roots)

    def _block(self, value: Any) -> None:
        if self._is_target(value):
            path = str(Path(os.fsdecode(value)).expanduser().resolve())
            self._attempts.append(path)
            raise TargetAccessError(f"target image access during preparation: {path}")

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        if hasattr(owner, name):
            original = getattr(owner, name)
            self._patches.append((owner, name, original))
            setattr(owner, name, replacement)

    def __enter__(self) -> "TargetDecodeSentinel":
        original_builtin_open = builtins.open
        original_io_open = io.open

        def guarded_builtin_open(file, *args, **kwargs):
            self._block(file)
            return original_builtin_open(file, *args, **kwargs)

        def guarded_io_open(file, *args, **kwargs):
            self._block(file)
            return original_io_open(file, *args, **kwargs)

        self._patch(builtins, "open", guarded_builtin_open)
        self._patch(io, "open", guarded_io_open)

        try:
            from PIL import Image

            original_pil_open = Image.open

            def guarded_pil_open(fp, *args, **kwargs):
                self._block(fp)
                return original_pil_open(fp, *args, **kwargs)

            self._patch(Image, "open", guarded_pil_open)
        except ImportError:
            pass

        cv2 = sys.modules.get("cv2")
        if cv2 is not None and hasattr(cv2, "imread"):
            original_imread = cv2.imread

            def guarded_imread(filename, *args, **kwargs):
                self._block(filename)
                return original_imread(filename, *args, **kwargs)

            self._patch(cv2, "imread", guarded_imread)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        while self._patches:
            owner, name, original = self._patches.pop()
            setattr(owner, name, original)

    def assert_clean(self) -> None:
        if self._attempts:
            raise TargetAccessError(
                f"target-access sentinel recorded {len(self._attempts)} blocked attempt(s)"
            )

    def manifest_record(self) -> dict[str, Any]:
        self.assert_clean()
        return {
            "enabled": True,
            "blocked_attempts": 0,
            "image_roots": [str(root) for root in self._roots],
        }


def artifact_identities(
    artifacts: Mapping[str, os.PathLike[str] | str]
) -> dict[str, dict[str, Any]]:
    identities = {}
    for name in sorted(artifacts):
        if not isinstance(name, str) or not name:
            raise ValueError("artifact logical names must be nonempty strings")
        path = Path(artifacts[name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"freeze artifact is not a file: {path}")
        identities[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return identities


def verify_artifact_identities(
    expected: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, os.PathLike[str] | str],
) -> dict[str, dict[str, Any]]:
    observed = artifact_identities(artifacts)
    if set(observed) != set(expected):
        raise IdentityDriftError("frozen artifact logical-name set changed")
    for name in sorted(observed):
        frozen = expected[name]
        if set(frozen) != {"bytes", "sha256"}:
            raise ValueError(f"malformed frozen identity for {name}")
        if observed[name] != {"bytes": int(frozen["bytes"]), "sha256": str(frozen["sha256"])}:
            raise IdentityDriftError(f"frozen artifact drift: {name}")
    return observed


def build_candidate_freeze_payload(
    *,
    source_revision: str,
    protocol_sha256: str,
    constants_sha256: str,
    preparation_artifacts: Mapping[str, os.PathLike[str] | str],
) -> dict[str, Any]:
    for label, value in (
        ("source_revision", source_revision),
        ("protocol_sha256", protocol_sha256),
        ("constants_sha256", constants_sha256),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a nonempty string")
    return {
        "schema": "gorfe-v1-candidate-freeze-v1",
        "source_revision": source_revision,
        "protocol_sha256": protocol_sha256,
        "constants_sha256": constants_sha256,
        "preparation_artifacts": artifact_identities(preparation_artifacts),
    }


def verify_candidate_freeze_payload(
    freeze: Mapping[str, Any],
    *,
    source_revision: str,
    protocol_sha256: str,
    constants_sha256: str,
    preparation_artifacts: Mapping[str, os.PathLike[str] | str],
) -> None:
    expected_header = {
        "schema": "gorfe-v1-candidate-freeze-v1",
        "source_revision": source_revision,
        "protocol_sha256": protocol_sha256,
        "constants_sha256": constants_sha256,
    }
    for key, expected in expected_header.items():
        if freeze.get(key) != expected:
            raise IdentityDriftError(f"candidate freeze {key} changed")
    identities = freeze.get("preparation_artifacts")
    if not isinstance(identities, Mapping):
        raise ValueError("candidate freeze lacks preparation_artifacts")
    verify_artifact_identities(identities, preparation_artifacts)
