"""Single-scene target-free preparation for the preregistered GoRFE-V1 gate.

The preparation transaction has one deliberately narrow responsibility: seal
the candidate topology and target-free renderer statistics.  It never opens an
image and never constructs ``Scene``.  All expensive work starts only after the
metadata, checkpoint, and runtime have passed their fail-closed preflight.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from gorfe_v1_evaluate_core import derive_eligibility
from gorfe_v1_io import (
    artifact_records,
    directory_bytes,
    save_torch_new,
    sha256_file,
    source_revision,
    write_json_new,
    write_sha256s,
    write_text_new,
)
from gorfe_v1_prepare_core import (
    IdentityDriftError,
    TargetAccessError,
    TargetDecodeSentinel,
    build_candidate_topology,
    load_validated_checkpoint,
    read_colmap_metadata,
    validate_scene_camera_identity,
)
from gorfe_v1_resources import (
    PhysicalGpuMonitor,
    ResourceMonitorError,
    host_peak_rss_bytes,
)
from gorfe_v1_scene import load_frozen_triangle_model, make_target_free_minicam
from gorfe_v1_state import CANDIDATE_MANIFEST_SCHEMA
from gorfe_v1_state import SCHEMA as CANDIDATE_STATE_SCHEMA
from gorfe_v1_state import candidate_state_payload
from gorfe_v1_stream import GoRFEV1Accumulator


REPOSITORY_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = REPOSITORY_ROOT / "experiments" / "gorfe_v1" / "protocol.md"
CONSTANTS_PATH = (
    REPOSITORY_ROOT / "experiments" / "gorfe_v1" / "protocol_constants.json"
)
PREPARE_RESULT_SCHEMA = "gorfe-v1-prepare-result-v1"


class PrepareInvalidError(RuntimeError):
    """A platform or resource condition invalidated the preparation attempt."""


def _load_constants() -> dict[str, Any]:
    with CONSTANTS_PATH.open("r", encoding="utf-8") as handle:
        constants = json.load(handle)
    required = {
        "camera_folds",
        "candidate_cap_edges",
        "candidate_hash_seed_uint64",
        "checkpoint_iteration",
        "checkpoint_sigma",
        "checkpoint_sigma_abs_tolerance",
        "checkpoint_texel_order",
        "checkpoint_vertex_sh_degree",
        "render_scaling",
        "resource_limits",
        "scene_names",
        "torch_cuda_build",
        "torch_version",
    }
    missing = sorted(required - set(constants))
    if missing:
        raise ValueError(f"protocol constants are missing: {', '.join(missing)}")
    return constants


def _tracked_checkout_is_clean() -> bool:
    for arguments in (
        ("git", "diff", "--quiet", "--"),
        ("git", "diff", "--cached", "--quiet", "--"),
    ):
        result = subprocess.run(
            arguments,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 1:
            return False
        if result.returncode != 0:
            raise RuntimeError(f"Git checkout preflight failed: {' '.join(arguments)}")
    return True


def _physical_gpu_preflight(physical_gpu: int) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise PrepareInvalidError(f"unexpected nvidia-smi GPU row: {output!r}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4 or any(not value.isdigit() for value in fields[1:]):
        raise PrepareInvalidError(f"malformed nvidia-smi GPU row: {rows[0]!r}")

    process_output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    processes = [row.strip() for row in process_output.splitlines() if row.strip()]
    if processes:
        raise PrepareInvalidError(
            f"physical GPU {physical_gpu} is not exclusive: {processes!r}"
        )
    return {
        "physical_gpu": physical_gpu,
        "name": fields[0],
        "total_mib": int(fields[1]),
        "used_mib": int(fields[2]),
        "free_mib": int(fields[3]),
        "compute_processes_before_cuda_init": [],
    }


def _native_extension_identity() -> dict[str, Any]:
    from diff_triangle_rasterization import _C

    path = Path(_C.__file__).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"native rasterizer extension is not a file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _runtime_preflight(
    physical_gpu: int,
    constants: Mapping[str, Any],
    *,
    initialize_cuda: bool = True,
) -> dict[str, Any]:
    if not isinstance(physical_gpu, int) or isinstance(physical_gpu, bool) or physical_gpu < 0:
        raise ValueError("physical_gpu must be a nonnegative integer")
    if torch.__version__ != str(constants["torch_version"]):
        raise PrepareInvalidError(
            f"PyTorch build drift: {torch.__version__} != {constants['torch_version']}"
        )
    if torch.version.cuda != str(constants["torch_cuda_build"]):
        raise PrepareInvalidError(
            f"CUDA build drift: {torch.version.cuda} != {constants['torch_cuda_build']}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise PrepareInvalidError(
            "CUDA_VISIBLE_DEVICES must expose exactly the requested physical GPU: "
            f"{visible!r} != {str(physical_gpu)!r}"
        )
    if not _tracked_checkout_is_clean():
        raise PrepareInvalidError("tracked checkout is dirty")

    # Query exclusivity before this process creates its own CUDA context.
    hardware = _physical_gpu_preflight(physical_gpu)
    limits = constants["resource_limits"]
    if hardware["name"] != "NVIDIA A40":
        raise PrepareInvalidError(f"GoRFE-V1 requires NVIDIA A40, got {hardware['name']!r}")
    if hardware["free_mib"] < int(limits["gpu_free_mib_at_start"]):
        raise PrepareInvalidError(
            f"physical GPU has only {hardware['free_mib']} MiB free at start"
        )
    if initialize_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise PrepareInvalidError("GoRFE-V1 requires exactly one visible CUDA device")
        if torch.cuda.current_device() != 0:
            raise PrepareInvalidError("the sole visible CUDA device must be logical cuda:0")
        logical_name = torch.cuda.get_device_name(0)
        if logical_name != "NVIDIA A40" or logical_name != hardware["name"]:
            raise PrepareInvalidError("logical and physical GPU identities disagree")

    return {
        "source_revision": source_revision(),
        "tracked_checkout_clean": True,
        "logical_device": "cuda:0",
        "cuda_context_initialized": bool(initialize_cuda),
        "cuda_visible_devices": visible,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "hardware_start": hardware,
        # Metadata/checkpoint-only preflight must not import a possibly stale or
        # not-yet-built native module.  The formal preparation path binds the
        # exact installed extension after the runner builds it.
        "native_extension": _native_extension_identity() if initialize_cuda else None,
    }


def _metadata_file_records(dataset_root: Path, metadata_format: str) -> dict[str, Any]:
    suffix = {"colmap_binary": "bin", "colmap_text": "txt"}.get(metadata_format)
    if suffix is None:
        raise ValueError(f"unknown COLMAP metadata format: {metadata_format!r}")
    sparse = dataset_root / "sparse" / "0"
    return artifact_records(
        {
            "cameras": sparse / f"cameras.{suffix}",
            "images": sparse / f"images.{suffix}",
        }
    )


def _verify_file_records(records: Mapping[str, Mapping[str, Any]]) -> None:
    for name, record in records.items():
        path = Path(str(record["path"]))
        observed = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        expected = {"bytes": int(record["bytes"]), "sha256": str(record["sha256"])}
        if observed != expected:
            raise IdentityDriftError(f"input file drifted during preparation: {name}")


def _verify_runtime_identity(runtime: Mapping[str, Any]) -> None:
    if source_revision() != runtime["source_revision"]:
        raise IdentityDriftError("source revision changed during preparation")
    if not _tracked_checkout_is_clean():
        raise IdentityDriftError("tracked source changed during preparation")
    _verify_file_records({"native_extension": runtime["native_extension"]})


def _sealed_protocol_records() -> dict[str, Any]:
    return artifact_records({"constants": CONSTANTS_PATH, "protocol": PROTOCOL_PATH})


def _render_design(view, model, background, face_edge_ids, candidate_count):
    from triangle_renderer import render

    pipe = SimpleNamespace(
        debug=False,
        convert_SHs_python=False,
        texel_footprint_filter=False,
    )
    return render(
        view,
        model,
        pipe,
        background,
        gorfe_face_edge_ids=face_edge_ids,
        gorfe_edge_count=candidate_count,
    )


def _normalise_exporter_diagnostics(value: Any) -> dict[str, int | float | bool | str]:
    if not isinstance(value, Mapping):
        raise TypeError("native exporter diagnostics must be a mapping")
    result: dict[str, int | float | bool | str] = {}
    for key, item in sorted(value.items()):
        if not isinstance(key, str) or not key:
            raise TypeError("native exporter diagnostic keys must be nonempty strings")
        if torch.is_tensor(item):
            if item.numel() != 1:
                raise TypeError(f"native exporter diagnostic {key!r} must be scalar")
            item = item.detach().cpu().item()
        if isinstance(item, (bool, int, float, str)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"native exporter diagnostic {key!r} is nonfinite")
            result[key] = item
        else:
            raise TypeError(f"native exporter diagnostic {key!r} is not JSON scalar")
    return result


def _copy_design_to_cpu(
    design: Mapping[str, Any],
    *,
    logical_device: torch.device,
    raw_row_limit: int,
):
    required = {"pixel_ids", "group_ids", "features", "diagnostics"}
    if not isinstance(design, Mapping) or set(design) != required:
        raise ValueError(f"GoRFE design fields must be exactly {sorted(required)}")
    pixel_ids = design["pixel_ids"]
    group_ids = design["group_ids"]
    features = design["features"]
    if not all(torch.is_tensor(value) for value in (pixel_ids, group_ids, features)):
        raise TypeError("GoRFE design rows must be tensors")
    if pixel_ids.ndim != 1 or group_ids.shape != pixel_ids.shape:
        raise ValueError("GoRFE pixel/group ids must be equal-length vectors")
    if features.shape != (pixel_ids.numel(), 4):
        raise ValueError("GoRFE features must have shape [raw rows, 4]")
    if pixel_ids.numel() > raw_row_limit:
        raise OverflowError("camera raw-row limit exceeded before CPU transfer")
    if pixel_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("GoRFE pixel ids must use int32 or int64")
    if group_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("GoRFE group ids must use int32 or int64")
    if features.dtype != torch.float32:
        raise TypeError("native GoRFE features must use float32")
    if {pixel_ids.device, group_ids.device, features.device} != {logical_device}:
        raise ValueError("native GoRFE rows must be on the sole logical render device")
    diagnostics = _normalise_exporter_diagnostics(design["diagnostics"])
    copied = (
        pixel_ids.detach().to(device="cpu", dtype=torch.int64).contiguous(),
        group_ids.detach().to(device="cpu", dtype=torch.int64).contiguous(),
        features.detach().to(device="cpu", dtype=torch.float32).contiguous(),
    )
    if logical_device.type == "cuda":
        torch.cuda.synchronize(logical_device)
    return (*copied, diagnostics)


def _carrier_summary(mask: torch.Tensor) -> dict[str, Any]:
    packed = mask.to(torch.uint8).contiguous().numpy().tobytes()
    import hashlib

    return {
        "eligible_count": int(mask.sum()),
        "mask_sha256": hashlib.sha256(packed).hexdigest(),
    }


def _eligibility_summary(eligibility) -> dict[str, Any]:
    dc = _carrier_summary(eligibility.dc)
    sh1 = _carrier_summary(eligibility.sh1)
    return {
        "dc": dc,
        "sh1": sh1,
        "both_count": int((eligibility.dc & eligibility.sh1).sum()),
    }


def _topology_summary(topology, *, face_count: int, cap: int, seed: int) -> dict[str, Any]:
    incidents = topology.candidate_incident_face_counts
    values, counts = torch.unique(incidents, sorted=True, return_counts=True)
    incidence_histogram = {
        str(int(value)): int(count) for value, count in zip(values, counts)
    }
    return {
        "face_count": int(face_count),
        "edge_count": int(topology.edge_count),
        "candidate_count": int(topology.candidate_edges.shape[0]),
        "candidate_cap": int(cap),
        "candidate_seed_uint64": int(seed),
        "vertex_stride": int(topology.vertex_stride),
        "candidate_incidence_histogram": incidence_histogram,
        "incidence_one_candidate_count": int((incidents == 1).sum()),
        "incidence_two_candidate_count": int((incidents == 2).sum()),
        "incidence_greater_than_two_candidate_count": int((incidents > 2).sum()),
        "maximum_candidate_incidence": int(incidents.max()) if incidents.numel() else 0,
        "mapped_face_local_slots": int((topology.face_candidate_edges >= 0).sum()),
    }


def _cuda_peak_record(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":  # CPU-only test seam; CLI preflight can never select it.
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256s_size(output: Path) -> int:
    return sum(
        64 + 2 + len(str((output / name).resolve()).encode("utf-8")) + 1
        for name in ("candidate_state.pt", "candidate_manifest.json", "result.json")
    )


def _project_artifact_bytes(
    *,
    output: Path,
    state_bytes: int,
    manifest: dict[str, Any],
    result: dict[str, Any],
    terminal_text: str,
) -> int:
    estimate = 0
    for _ in range(16):
        manifest["resources"]["persistent_artifact_bytes"] = estimate
        result["resources"]["persistent_artifact_bytes"] = estimate
        manifest["resources"]["measurements"]["persistent_artifact_bytes"] = estimate
        result["resources"]["measurements"]["persistent_artifact_bytes"] = estimate
        updated = (
            state_bytes
            + len(_json_bytes(manifest))
            + len(_json_bytes(result))
            + _sha256s_size(output)
            + len(terminal_text.encode("utf-8"))
        )
        if updated == estimate:
            return updated
        estimate = updated
    raise RuntimeError("persistent artifact byte projection did not converge")


def _resource_record(
    *,
    limits: Mapping[str, Any],
    wall_seconds: float,
    cuda_peak: Mapping[str, int],
    physical_gpu: Mapping[str, Any],
    host_peak_bytes: int,
    maximum_raw_rows: int,
) -> dict[str, Any]:
    measurements = {
        "wall_seconds_through_candidate_state": float(wall_seconds),
        "cuda_peak_allocated_bytes": int(cuda_peak["allocated_bytes"]),
        "cuda_peak_reserved_bytes": int(cuda_peak["reserved_bytes"]),
        "physical_gpu_peak_used_mib": int(physical_gpu["peak_used_mib"]),
        "physical_gpu_minimum_free_mib": int(physical_gpu["minimum_free_mib"]),
        "host_peak_rss_bytes": int(host_peak_bytes),
        "maximum_raw_rows_per_camera": int(maximum_raw_rows),
        "persistent_artifact_bytes": 0,
    }
    checks = {
        "wall_seconds": measurements["wall_seconds_through_candidate_state"]
        <= int(limits["wall_seconds_per_scene"]),
        "cuda_allocated_bytes": measurements["cuda_peak_allocated_bytes"]
        <= int(limits["cuda_allocated_bytes"]),
        "physical_gpu_used_mib": measurements["physical_gpu_peak_used_mib"]
        <= int(limits["physical_gpu_used_mib"]),
        "host_peak_rss_bytes": measurements["host_peak_rss_bytes"]
        <= int(limits["host_peak_rss_bytes"]),
        "raw_rows_per_camera": measurements["maximum_raw_rows_per_camera"]
        <= int(limits["raw_rows_per_camera"]),
        "persistent_artifact_bytes": True,
    }
    return {
        "limits": {name: int(value) for name, value in sorted(limits.items())},
        "measurements": measurements,
        "physical_gpu_monitor": dict(physical_gpu),
        "checks": checks,
        "persistent_artifact_bytes": 0,
        "violations": sorted(name for name, passed in checks.items() if not passed),
    }


def _reserve_output(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)


def _terminal_error(output: Path, name: str, error: BaseException) -> None:
    marker = output / name
    if marker.exists() or (output / "DONE").exists():
        return
    write_text_new(marker, f"{type(error).__name__}: {error}\n")


def _prepare_after_preflight(
    *,
    scene: str,
    dataset_root: Path,
    model_root: Path,
    output: Path,
    constants: Mapping[str, Any],
    split,
    dataset_identity: Mapping[str, Any],
    metadata_files: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
    checkpoint_inspection,
    protocol_files: Mapping[str, Any],
    runtime: Mapping[str, Any],
    sentinel: TargetDecodeSentinel,
    invocation_start: float,
) -> dict[str, Any]:
    device = torch.device(str(runtime["logical_device"]))
    limits = constants["resource_limits"]
    cap = int(constants["candidate_cap_edges"])
    seed = int(constants["candidate_hash_seed_uint64"])
    fold_count = int(constants["camera_folds"])
    raw_limit = int(limits["raw_rows_per_camera"])
    phase_times: dict[str, float] = {}
    camera_records = []

    _reset_cuda_peak(device)
    monitor = PhysicalGpuMonitor(
        int(runtime["hardware_start"]["physical_gpu"]), exclusive_pid=os.getpid()
    )
    with monitor:
        started = time.monotonic()
        topology = build_candidate_topology(
            checkpoint_state["_triangle_indices"], seed=seed, cap=cap
        )
        phase_times["topology_seconds"] = time.monotonic() - started
        candidate_count = int(topology.candidate_edges.shape[0])
        if candidate_count < 1:
            raise ValueError("the frozen mesh produced no candidate edge")

        started = time.monotonic()
        model = load_frozen_triangle_model(checkpoint_state, device=device)
        if (
            model.scaling != int(constants["render_scaling"])
            or model.active_sh_degree != int(constants["checkpoint_vertex_sh_degree"])
            or model.texel_order != int(constants["checkpoint_texel_order"])
            or model.optimizer is not None
            or model.texel_optimizer is not None
        ):
            raise ValueError("frozen renderer model violates the GoRFE-V1 contract")
        face_edge_ids = topology.face_candidate_edges.to(
            device=device, dtype=torch.int32
        ).contiguous()
        background = torch.zeros(3, dtype=torch.float32, device=device)
        accumulator = GoRFEV1Accumulator(
            candidate_count,
            fold_count=fold_count,
            device="cpu",
            raw_row_limit=raw_limit,
        )
        phase_times["model_and_accumulator_seconds"] = time.monotonic() - started

        started = time.monotonic()
        for uid, metadata in enumerate(split.train_cameras):
            view = make_target_free_minicam(metadata, uid=uid, device=device)
            with torch.no_grad():
                rendered = _render_design(
                    view, model, background, face_edge_ids, candidate_count
                )
            if not isinstance(rendered, Mapping) or "gorfe_design" not in rendered:
                raise ValueError("renderer did not return a GoRFE design")
            pixel_ids, group_ids, features, exporter = _copy_design_to_cpu(
                rendered["gorfe_design"],
                logical_device=device,
                raw_row_limit=raw_limit,
            )
            # No GPU result from one camera may survive into the next reduction.
            del rendered
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            reduction = accumulator.add_camera(
                name=metadata.image_name,
                fold=metadata.fold,
                pixel_count=int(view.image_width * view.image_height),
                pixel_ids=pixel_ids,
                group_ids=group_ids,
                features=features,
                residuals=None,
            )
            duplicate_rows = reduction.input_rows - reduction.reduced_rows
            camera_records.append(
                {
                    "name": metadata.image_name,
                    "fold": int(metadata.fold),
                    "output_width": int(view.image_width),
                    "output_height": int(view.image_height),
                    "exporter": exporter,
                    "reduction": {
                        **asdict(reduction),
                        "duplicate_rows": int(duplicate_rows),
                        "duplicate_fraction": (
                            float(duplicate_rows / reduction.input_rows)
                            if reduction.input_rows
                            else 0.0
                        ),
                    },
                }
            )
            del view, pixel_ids, group_ids, features
        phase_times["camera_export_and_reduction_seconds"] = time.monotonic() - started

        started = time.monotonic()
        statistics = accumulator.statistics()
        eligibility = derive_eligibility(statistics)
        phase_times["eligibility_seconds"] = time.monotonic() - started
        cuda_peak = _cuda_peak_record(device)
        del model, face_edge_ids, background
        _empty_cuda_cache(device)

    physical_record = monitor.record()
    topology_record = _topology_summary(
        topology,
        face_count=checkpoint_inspection.face_count,
        cap=cap,
        seed=seed,
    )
    eligibility_record = _eligibility_summary(eligibility)

    # Recheck every file identity after native export and before sealing state.
    _verify_file_records(metadata_files)
    _verify_file_records(
        {
            "checkpoint": {
                "path": checkpoint_inspection.path,
                "bytes": checkpoint_inspection.bytes,
                "sha256": checkpoint_inspection.sha256,
            }
        }
    )
    _verify_file_records(protocol_files)
    _verify_runtime_identity(runtime)
    sentinel_record = sentinel.manifest_record()

    state_path = output / "candidate_state.pt"
    started = time.monotonic()
    save_torch_new(
        state_path,
        candidate_state_payload(
            scene=scene,
            topology=topology,
            statistics=statistics,
            eligibility=eligibility,
        ),
    )
    phase_times["candidate_state_write_seconds"] = time.monotonic() - started
    state_identity = artifact_records({"candidate_state": state_path})["candidate_state"]

    wall_seconds = time.monotonic() - invocation_start
    maximum_raw_rows = max(
        (record["reduction"]["input_rows"] for record in camera_records), default=0
    )
    resources = _resource_record(
        limits=limits,
        wall_seconds=wall_seconds,
        cuda_peak=cuda_peak,
        physical_gpu=physical_record,
        host_peak_bytes=host_peak_rss_bytes(),
        maximum_raw_rows=maximum_raw_rows,
    )
    protocol_sha = str(protocol_files["protocol"]["sha256"])
    constants_sha = str(protocol_files["constants"]["sha256"])
    fold_map = split.fold_map
    manifest = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "experiment": "GoRFE-V1",
        "phase": "prepare",
        "scene": scene,
        "source_revision": runtime["source_revision"],
        "protocol_sha256": protocol_sha,
        "constants_sha256": constants_sha,
        "protocol_files": dict(protocol_files),
        "dataset": {
            "root": str(dataset_root),
            "image_root": str((dataset_root / "images").resolve()),
            "metadata_format": split.metadata_format,
            "metadata_files": metadata_files,
            "identity": dict(dataset_identity),
            "fold": {
                "count": fold_count,
                "sizes": list(split.fold_sizes),
                "map_sha256": split.fold_map_sha256,
                "name_to_fold": fold_map,
            },
        },
        "checkpoint": {
            "model_root": str(model_root),
            **checkpoint_inspection.to_manifest(),
        },
        "runtime": dict(runtime),
        "render_contract": {
            "background": "black",
            "scaling": int(constants["render_scaling"]),
            "native_sh": True,
            "texels": False,
            "window_donors": False,
            "frozen_checkpoint": True,
        },
        "target_decode_sentinel": sentinel_record,
        "topology": topology_record,
        "eligibility": eligibility_record,
        "candidate_state_schema": CANDIDATE_STATE_SCHEMA,
        "candidate_state": state_identity,
        "cameras": camera_records,
        "phase_times": phase_times,
        "resources": resources,
    }
    result = {
        "schema": PREPARE_RESULT_SCHEMA,
        "experiment": "GoRFE-V1",
        "phase": "prepare",
        "scene": scene,
        "decision": "prepared",
        "candidate_count": candidate_count,
        "eligibility": eligibility_record,
        "candidate_state": {
            "bytes": state_identity["bytes"],
            "sha256": state_identity["sha256"],
        },
        "resources": resources.copy(),
    }
    result["resources"] = {
        **resources,
        "measurements": dict(resources["measurements"]),
        "checks": dict(resources["checks"]),
        "violations": list(resources["violations"]),
    }

    non_artifact_violations = list(resources["violations"])
    prepared_terminal = "complete\n"
    prepared_bytes = _project_artifact_bytes(
        output=output,
        state_bytes=int(state_identity["bytes"]),
        manifest=manifest,
        result=result,
        terminal_text=prepared_terminal,
    )
    artifact_limit = int(limits["artifact_bytes_per_scene"])
    if prepared_bytes > artifact_limit:
        non_artifact_violations.append("persistent_artifact_bytes")

    if non_artifact_violations:
        decision = "invalid"
        violations = sorted(set(non_artifact_violations))
        terminal_name = "INVALID"
        terminal_text = "resource limits exceeded: " + ", ".join(violations) + "\n"
    else:
        decision = "prepared"
        violations = []
        terminal_name = "DONE"
        terminal_text = prepared_terminal

    result["decision"] = decision
    for record in (manifest["resources"], result["resources"]):
        record["checks"]["persistent_artifact_bytes"] = (
            "persistent_artifact_bytes" not in violations
        )
        record["violations"] = violations
    final_bytes = _project_artifact_bytes(
        output=output,
        state_bytes=int(state_identity["bytes"]),
        manifest=manifest,
        result=result,
        terminal_text=terminal_text,
    )
    if final_bytes > artifact_limit and "persistent_artifact_bytes" not in violations:
        violations = sorted((*violations, "persistent_artifact_bytes"))
        decision = "invalid"
        terminal_name = "INVALID"
        terminal_text = "resource limits exceeded: " + ", ".join(violations) + "\n"
        result["decision"] = decision
        for record in (manifest["resources"], result["resources"]):
            record["checks"]["persistent_artifact_bytes"] = False
            record["violations"] = violations
        final_bytes = _project_artifact_bytes(
            output=output,
            state_bytes=int(state_identity["bytes"]),
            manifest=manifest,
            result=result,
            terminal_text=terminal_text,
        )
    for record in (manifest["resources"], result["resources"]):
        record["persistent_artifact_bytes"] = final_bytes
        record["measurements"]["persistent_artifact_bytes"] = final_bytes

    manifest_path = write_json_new(output / "candidate_manifest.json", manifest)
    result_path = write_json_new(output / "result.json", result)
    write_sha256s(output / "SHA256SUMS", (state_path, manifest_path, result_path))
    actual_before_terminal = directory_bytes(output)
    if actual_before_terminal + len(terminal_text.encode("utf-8")) != final_bytes:
        raise RuntimeError("persistent artifact byte projection disagrees with disk")
    write_text_new(output / terminal_name, terminal_text)
    return result


def _collect_preflight(
    *,
    scene: str,
    dataset_root: Path,
    model_root: Path,
    physical_gpu: int,
    initialize_cuda: bool,
) -> dict[str, Any]:
    constants = _load_constants()
    if scene not in constants["scene_names"]:
        raise ValueError(f"scene must be one of {constants['scene_names']}")
    split = read_colmap_metadata(
        dataset_root, fold_count=int(constants["camera_folds"])
    )
    dataset_identity = validate_scene_camera_identity(scene, split, constants)
    metadata_files = _metadata_file_records(dataset_root, split.metadata_format)
    checkpoint_state, checkpoint_inspection = load_validated_checkpoint(
        model_root,
        iteration=int(constants["checkpoint_iteration"]),
        expected_sh_degree=int(constants["checkpoint_vertex_sh_degree"]),
        expected_texel_order=int(constants["checkpoint_texel_order"]),
        expected_sigma=float(constants["checkpoint_sigma"]),
        sigma_abs_tolerance=float(constants["checkpoint_sigma_abs_tolerance"]),
    )
    protocol_files = _sealed_protocol_records()
    runtime = _runtime_preflight(
        physical_gpu, constants, initialize_cuda=initialize_cuda
    )
    return {
        "constants": constants,
        "split": split,
        "dataset_identity": dataset_identity,
        "metadata_files": metadata_files,
        "checkpoint_state": checkpoint_state,
        "checkpoint_inspection": checkpoint_inspection,
        "protocol_files": protocol_files,
        "runtime": runtime,
    }


def preflight_scene(
    *,
    scene: str,
    dataset_root: os.PathLike[str] | str,
    model_root: os.PathLike[str] | str,
    physical_gpu: int,
) -> dict[str, Any]:
    """Validate one scene without creating output, topology, or a CUDA context."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    model_root = Path(model_root).expanduser().resolve()
    with TargetDecodeSentinel([dataset_root / "images"]) as sentinel:
        prepared = _collect_preflight(
            scene=scene,
            dataset_root=dataset_root,
            model_root=model_root,
            physical_gpu=physical_gpu,
            initialize_cuda=False,
        )
        sentinel_record = sentinel.manifest_record()
    split = prepared["split"]
    inspection = prepared["checkpoint_inspection"]
    return {
        "schema": "gorfe-v1-prepare-preflight-v1",
        "phase": "prepare-preflight",
        "scene": scene,
        "dataset": {
            "root": str(dataset_root),
            "metadata_format": split.metadata_format,
            "metadata_files": prepared["metadata_files"],
            "identity": prepared["dataset_identity"],
            "fold": {
                "count": int(prepared["constants"]["camera_folds"]),
                "sizes": list(split.fold_sizes),
                "map_sha256": split.fold_map_sha256,
            },
        },
        "checkpoint": {"model_root": str(model_root), **inspection.to_manifest()},
        "protocol_files": prepared["protocol_files"],
        "runtime": prepared["runtime"],
        "target_decode_sentinel": sentinel_record,
        "checks": {
            "metadata": True,
            "checkpoint": True,
            "runtime_without_cuda_context": True,
            "target_decode_sentinel": True,
        },
    }


def prepare_scene(
    *,
    scene: str,
    dataset_root: os.PathLike[str] | str,
    model_root: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
    physical_gpu: int,
) -> dict[str, Any]:
    """Prepare one scene; return the written result or raise before/after FAILED."""
    invocation_start = time.monotonic()
    dataset_root = Path(dataset_root).expanduser().resolve()
    model_root = Path(model_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"GoRFE-V1 output already exists: {output}")
    image_root = dataset_root / "images"

    with TargetDecodeSentinel([image_root]) as sentinel:
        try:
            # Nothing below this line may create the attempt directory until all
            # three preflight domains have succeeded.
            preflight = _collect_preflight(
                scene=scene,
                dataset_root=dataset_root,
                model_root=model_root,
                physical_gpu=physical_gpu,
                initialize_cuda=True,
            )
            sentinel.assert_clean()
            _reserve_output(output)
            return _prepare_after_preflight(
                scene=scene,
                dataset_root=dataset_root,
                model_root=model_root,
                output=output,
                constants=preflight["constants"],
                split=preflight["split"],
                dataset_identity=preflight["dataset_identity"],
                metadata_files=preflight["metadata_files"],
                checkpoint_state=preflight["checkpoint_state"],
                checkpoint_inspection=preflight["checkpoint_inspection"],
                protocol_files=preflight["protocol_files"],
                runtime=preflight["runtime"],
                sentinel=sentinel,
                invocation_start=invocation_start,
            )
        except TargetAccessError as error:
            if output.is_dir():
                _terminal_error(output, "INVALID", error)
            raise
        except (PrepareInvalidError, ResourceMonitorError, OverflowError) as error:
            if output.is_dir():
                _terminal_error(output, "INVALID", error)
            raise
        except Exception as error:
            if output.is_dir():
                _terminal_error(output, "FAILED", error)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("garden", "room"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        result = preflight_scene(
            scene=args.scene,
            dataset_root=args.dataset_root,
            model_root=args.model_root,
            physical_gpu=args.physical_gpu,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required unless --preflight-only is used")
    result = prepare_scene(
        scene=args.scene,
        dataset_root=args.dataset_root,
        model_root=args.model_root,
        output=args.output,
        physical_gpu=args.physical_gpu,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "prepared" else 2


if __name__ == "__main__":
    raise SystemExit(main())
