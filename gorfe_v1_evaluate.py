"""Evaluate one frozen GoRFE-V1 scene after the candidate-freeze commit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import torch

from gorfe_v1 import PROTOCOL_CONSTANTS
from gorfe_v1_evaluate_core import SceneInvalidError, derive_eligibility, evaluate_scene
from gorfe_v1_freeze_payload import preparation_artifact_paths
from gorfe_v1_io import (
    artifact_records,
    directory_bytes,
    save_torch_new,
    sha256_file,
    write_json_new,
    write_sha256s,
    write_text_new,
)
from gorfe_v1_prepare_core import (
    IdentityDriftError,
    OfficialSplitGuard,
    read_colmap_metadata,
    validate_scene_camera_identity,
    verify_candidate_freeze_payload,
)
from gorfe_v1_resources import (
    PhysicalGpuMonitor,
    ResourceMonitorError,
    host_peak_rss_bytes,
    query_gpu_compute_processes,
    query_gpu_memory,
)
from gorfe_v1_scene import load_frozen_triangle_model, load_training_rgb, make_target_free_minicam
from gorfe_v1_state import (
    CANDIDATE_MANIFEST_SCHEMA,
    load_candidate_state,
    stream_to_payload,
)
from gorfe_v1_stream import GoRFEV1Accumulator
from gorfe_v1_prepare_core import load_validated_checkpoint


REPOSITORY = Path(__file__).resolve().parent
PROTOCOL_PATH = REPOSITORY / "experiments" / "gorfe_v1" / "protocol.md"
CONSTANTS_PATH = REPOSITORY / "experiments" / "gorfe_v1" / "protocol_constants.json"
EVALUATION_STATE_SCHEMA = "gorfe-v1-evaluation-state-v1"


class ResourceLimitError(RuntimeError):
    """A valid scientific decision cannot be issued within the locked limits."""


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def verify_tracked_freeze_commit(freeze_path: Path, implementation_revision: str) -> str:
    """Require a clean commit whose sole delta is the supplied freeze JSON."""
    freeze_path = freeze_path.resolve()
    try:
        relative = freeze_path.relative_to(REPOSITORY).as_posix()
    except ValueError as error:
        raise IdentityDriftError("candidate freeze must live inside the repository") from error
    if not relative.startswith("experiments/gorfe_v1/candidate_freeze_") or not relative.endswith(
        ".json"
    ):
        raise IdentityDriftError("candidate freeze has an unexpected tracked path")
    _git("ls-files", "--error-unmatch", relative)
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise IdentityDriftError("tracked checkout is dirty")
    current = _git("rev-parse", "HEAD")
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", implementation_revision, current],
            cwd=REPOSITORY,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as error:
        raise IdentityDriftError("freeze implementation revision is not an ancestor") from error
    changed = [
        row
        for row in _git("diff", "--name-only", f"{implementation_revision}..{current}").splitlines()
        if row
    ]
    if changed != [relative]:
        raise IdentityDriftError(
            "evaluation commit may differ from the implementation only by its freeze JSON"
        )
    return current


def _metadata_files(dataset_root: Path) -> dict[str, dict]:
    sparse = dataset_root / "sparse" / "0"
    if (sparse / "images.bin").is_file() and (sparse / "cameras.bin").is_file():
        paths = {"images": sparse / "images.bin", "cameras": sparse / "cameras.bin"}
    elif (sparse / "images.txt").is_file() and (sparse / "cameras.txt").is_file():
        paths = {"images": sparse / "images.txt", "cameras": sparse / "cameras.txt"}
    else:
        raise FileNotFoundError(f"complete COLMAP metadata not found under {sparse}")
    return artifact_records(paths)


def _require_same_metadata(expected: dict, observed: dict) -> None:
    if set(expected) != set(observed):
        raise IdentityDriftError("COLMAP metadata file set changed after candidate freeze")
    for name in sorted(expected):
        frozen = expected[name]
        current = observed[name]
        if (
            frozen.get("sha256") != current["sha256"]
            or int(frozen.get("bytes", -1)) != current["bytes"]
        ):
            raise IdentityDriftError(f"COLMAP metadata changed after freeze: {name}")


def _runtime_preflight(physical_gpu: int) -> dict:
    if torch.__version__ != PROTOCOL_CONSTANTS["torch_version"]:
        raise IdentityDriftError(
            f"torch build drift: {torch.__version__} != {PROTOCOL_CONSTANTS['torch_version']}"
        )
    if torch.version.cuda != PROTOCOL_CONSTANTS["torch_cuda_build"]:
        raise IdentityDriftError(
            f"CUDA build drift: {torch.version.cuda} != {PROTOCOL_CONSTANTS['torch_cuda_build']}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise IdentityDriftError(
            f"CUDA_VISIBLE_DEVICES must be exactly {physical_gpu}, got {visible!r}"
        )
    processes = query_gpu_compute_processes(physical_gpu)
    if processes:
        raise ResourceLimitError(
            f"physical GPU {physical_gpu} is not exclusive: {list(processes)!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise IdentityDriftError("GoRFE-V1 requires exactly one visible CUDA device")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != "NVIDIA A40":
        raise IdentityDriftError(f"GoRFE-V1 requires NVIDIA A40, got {gpu_name}")
    memory = query_gpu_memory(physical_gpu)
    minimum = int(PROTOCOL_CONSTANTS["resource_limits"]["gpu_free_mib_at_start"])
    if memory["free_mib"] < minimum:
        raise ResourceLimitError(
            f"physical GPU {physical_gpu} has {memory['free_mib']} MiB free, need {minimum}"
        )
    return {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_visible_devices": visible,
        "physical_gpu": physical_gpu,
        "gpu": gpu_name,
        "compute_processes_before_cuda_init": list(processes),
        "start_memory": memory,
    }


def _extension_identity(prepare_root: Path) -> dict:
    from diff_triangle_rasterization import _C

    installed = Path(_C.__file__).resolve()
    frozen = prepare_root / "native_extension.so"
    installed_sha = sha256_file(installed)
    frozen_sha = sha256_file(frozen)
    if installed_sha != frozen_sha:
        raise IdentityDriftError("loaded native extension differs from the frozen binary")
    return {
        "loaded_path": str(installed),
        "loaded_sha256": installed_sha,
        "frozen_path": str(frozen.resolve()),
        "frozen_sha256": frozen_sha,
    }


def _target_path(image_root: Path, camera) -> Path:
    return image_root / Path(camera.relative_image_path).name


def _camera_record(metadata, camera, reduction, exporter, target_sha256: str) -> dict:
    input_rows = int(reduction.input_rows)
    reduced_rows = int(reduction.reduced_rows)
    return {
        "image_name": metadata.image_name,
        "fold": int(metadata.fold),
        "width": int(camera.image_width),
        "height": int(camera.image_height),
        "target_sha256": target_sha256,
        "input_rows": input_rows,
        "reduced_rows": reduced_rows,
        "duplicate_rows": input_rows - reduced_rows,
        "duplicate_fraction": (
            (input_rows - reduced_rows) / input_rows if input_rows else 0.0
        ),
        "dc_support_rows": int(reduction.dc_support_rows),
        "sh1_support_rows": int(reduction.sh1_support_rows),
        "exporter": {name: int(value) for name, value in sorted(exporter.items())},
    }


def _check_resources(resource_record: dict) -> None:
    limits = PROTOCOL_CONSTANTS["resource_limits"]
    checks = {
        "wall_seconds": resource_record["wall_seconds"] <= float(limits["wall_seconds_per_scene"]),
        "cuda_allocated_bytes": resource_record["cuda_peak_allocated_bytes"]
        <= int(limits["cuda_allocated_bytes"]),
        "physical_gpu_used_mib": resource_record["physical_gpu"]["peak_used_mib"]
        <= int(limits["physical_gpu_used_mib"]),
        "host_peak_rss_bytes": resource_record["host_peak_rss_bytes"]
        <= int(limits["host_peak_rss_bytes"]),
    }
    resource_record["checks"] = checks
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ResourceLimitError("resource ceiling exceeded: " + ", ".join(failed))


def evaluate_frozen_candidate_state(
    *, scene, candidate_state, target_free_statistics, evaluation_statistics, frozen_masks
):
    """Keep the hash-verified candidate state as the sole endpoint authority."""
    endpoints = candidate_state.get("candidate_edges")
    if not torch.is_tensor(endpoints):
        raise IdentityDriftError("verified candidate state has no endpoint tensor")
    return evaluate_scene(
        scene=scene,
        candidate_endpoints=endpoints,
        target_free_statistics=target_free_statistics,
        evaluation_statistics=evaluation_statistics,
        frozen_masks=frozen_masks,
    )


def _predicted_artifact_bytes(output: Path, manifest: dict, hashed_paths: list[Path]) -> int:
    fixed = sum(path.stat().st_size for path in hashed_paths)
    value = 0
    while True:
        manifest["resources"]["artifact_bytes"] = value
        manifest_bytes = len(
            (
                json.dumps(
                    manifest,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
        checksum_paths = [*hashed_paths, output / "manifest.json"]
        checksum_bytes = sum(
            64 + 2 + len(str(path.resolve()).encode("utf-8")) + 1
            for path in checksum_paths
        )
        updated = fixed + manifest_bytes + checksum_bytes + len("complete\n")
        if updated == value:
            return value
        value = updated


def run_scene(
    *,
    scene: str,
    dataset_root: Path,
    model_root: Path,
    prepare_root: Path,
    freeze_file: Path,
    output: Path,
    physical_gpu: int,
) -> dict:
    invocation_start = time.perf_counter()
    if scene not in PROTOCOL_CONSTANTS["scene_names"]:
        raise ValueError(f"unknown locked scene {scene!r}")
    dataset_root = dataset_root.resolve()
    model_root = model_root.resolve()
    prepare_root = prepare_root.resolve()
    freeze_file = freeze_file.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse evaluation directory: {output}")

    # This entire block precedes the first target path lookup or image decoder import.
    freeze = _load_json(freeze_file)
    implementation_revision = str(freeze.get("source_revision", ""))
    evaluation_revision = verify_tracked_freeze_commit(freeze_file, implementation_revision)
    verify_candidate_freeze_payload(
        freeze,
        source_revision=implementation_revision,
        protocol_sha256=sha256_file(PROTOCOL_PATH),
        constants_sha256=sha256_file(CONSTANTS_PATH),
        preparation_artifacts=preparation_artifact_paths(prepare_root),
    )
    candidate_manifest_path = prepare_root / scene / "candidate_manifest.json"
    candidate_state_path = prepare_root / scene / "candidate_state.pt"
    candidate_manifest = _load_json(candidate_manifest_path)
    if (
        candidate_manifest.get("schema") != CANDIDATE_MANIFEST_SCHEMA
        or candidate_manifest.get("scene") != scene
        or candidate_manifest.get("source_revision") != implementation_revision
    ):
        raise IdentityDriftError("candidate manifest header differs from the freeze")
    split = read_colmap_metadata(dataset_root, fold_count=int(PROTOCOL_CONSTANTS["camera_folds"]))
    camera_identity = validate_scene_camera_identity(scene, split, PROTOCOL_CONSTANTS)
    if candidate_manifest["dataset"]["identity"] != camera_identity:
        raise IdentityDriftError("camera split identity differs from candidate preparation")
    _require_same_metadata(
        candidate_manifest["dataset"]["metadata_files"], _metadata_files(dataset_root)
    )
    if Path(candidate_manifest["dataset"]["root"]).resolve() != dataset_root:
        raise IdentityDriftError("dataset root differs from candidate preparation")

    expected_checkpoint = candidate_manifest["checkpoint"]
    checkpoint_state, checkpoint = load_validated_checkpoint(
        model_root,
        iteration=int(PROTOCOL_CONSTANTS["checkpoint_iteration"]),
        expected_sha256=expected_checkpoint["sha256"],
        expected_sh_degree=int(PROTOCOL_CONSTANTS["checkpoint_vertex_sh_degree"]),
        expected_texel_order=int(PROTOCOL_CONSTANTS["checkpoint_texel_order"]),
        expected_sigma=float(PROTOCOL_CONSTANTS["checkpoint_sigma"]),
        sigma_abs_tolerance=float(PROTOCOL_CONSTANTS["checkpoint_sigma_abs_tolerance"]),
    )
    if checkpoint.to_manifest() != {
        key: expected_checkpoint[key] for key in checkpoint.to_manifest()
    }:
        raise IdentityDriftError("checkpoint inspection differs from candidate preparation")
    if Path(expected_checkpoint["model_root"]).resolve() != model_root:
        raise IdentityDriftError("model root differs from candidate preparation")
    state, target_free_statistics, frozen_masks = load_candidate_state(
        candidate_state_path,
        expected_scene=scene,
        expected_face_count=checkpoint.face_count,
    )
    recomputed_masks = derive_eligibility(target_free_statistics)
    if not torch.equal(recomputed_masks.dc, frozen_masks.dc) or not torch.equal(
        recomputed_masks.sh1, frozen_masks.sh1
    ):
        raise IdentityDriftError("candidate eligibility masks differ from frozen support")

    runtime = _runtime_preflight(physical_gpu)
    extension = _extension_identity(prepare_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    torch.cuda.reset_peak_memory_stats()
    camera_rows = []
    image_root = dataset_root / "images"
    split_guard = OfficialSplitGuard(split)
    accumulator = GoRFEV1Accumulator(
        int(state["candidate_edges"].shape[0]),
        fold_count=int(PROTOCOL_CONSTANTS["camera_folds"]),
        device="cpu",
        raw_row_limit=int(PROTOCOL_CONSTANTS["resource_limits"]["raw_rows_per_camera"]),
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(
        convert_SHs_python=False, debug=False, texel_footprint_filter=False
    )
    # Kept below all freeze checks so CPU preflight and tests do not import a
    # possibly stale native rasterizer, and no rendering can precede identity.
    from triangle_renderer import render as render_scene

    with PhysicalGpuMonitor(physical_gpu, exclusive_pid=os.getpid()) as monitor:
        model = load_frozen_triangle_model(checkpoint_state, device="cuda")
        candidate_map = state["face_candidate_edges"].to(
            device="cuda", dtype=torch.int32, copy=True
        ).contiguous()
        with torch.no_grad():
            for uid, metadata in enumerate(split.train_cameras):
                camera = make_target_free_minicam(metadata, uid=uid, device="cuda")
                package = render_scene(
                    camera,
                    model,
                    pipeline,
                    background,
                    gorfe_face_edge_ids=candidate_map,
                    gorfe_edge_count=int(state["candidate_edges"].shape[0]),
                )
                design = package["gorfe_design"]
                pixel_ids = design["pixel_ids"].detach().cpu()
                group_ids = design["group_ids"].detach().cpu()
                features = design["features"].detach().cpu()

                # The guard is deliberately called before hashing, path existence,
                # PIL import, or decode.  Only official training pixels enter here.
                split_guard.require_training(metadata.image_name)
                target_path = _target_path(image_root, metadata)
                target_sha = sha256_file(target_path)
                target = load_training_rgb(
                    metadata, split_guard=split_guard, image_root=image_root
                )
                parent = package["render"].detach().cpu()
                if parent.shape != target.shape or parent.dtype != torch.float32:
                    raise SceneInvalidError(
                        f"parent/target shape or dtype mismatch for {metadata.image_name}"
                    )
                residual = (target - parent).permute(1, 2, 0).reshape(-1, 3).to(torch.float64)
                reduction = accumulator.add_camera(
                    name=metadata.image_name,
                    fold=int(metadata.fold),
                    pixel_count=int(camera.image_height * camera.image_width),
                    pixel_ids=pixel_ids,
                    group_ids=group_ids,
                    features=features,
                    residuals=residual,
                )
                camera_rows.append(
                    _camera_record(
                        metadata, camera, reduction, design["diagnostics"], target_sha
                    )
                )
                del package, design, pixel_ids, group_ids, features, target, parent, residual, camera
        torch.cuda.synchronize()
    physical = monitor.record()
    evaluation_statistics = accumulator.statistics()
    scene_evaluation = evaluate_frozen_candidate_state(
        scene=scene,
        candidate_state=state,
        target_free_statistics=target_free_statistics,
        evaluation_statistics=evaluation_statistics,
        frozen_masks=frozen_masks,
    )

    evaluation_state = {
        "schema": EVALUATION_STATE_SCHEMA,
        "scene": scene,
        "statistics": stream_to_payload(evaluation_statistics),
        "eligible_type_ids": scene_evaluation.type_ids,
        "eligible_endpoints": scene_evaluation.endpoints,
    }
    state_path = save_torch_new(output / "evaluation_state.pt", evaluation_state)
    resource_record = {
        "wall_seconds": time.perf_counter() - invocation_start,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "physical_gpu": physical,
        "host_peak_rss_bytes": host_peak_rss_bytes(),
        "artifact_bytes": 0,
    }
    _check_resources(resource_record)

    result = {
        "experiment": "GoRFE-V1",
        "phase": "evaluate",
        "scene": scene,
        "decision": "pass" if scene_evaluation.result["passing_families"] else "fail",
        **scene_evaluation.result,
    }
    result_path = write_json_new(output / "result.json", result)
    manifest = {
        "schema": "gorfe-v1-evaluation-manifest-v1",
        "experiment": "GoRFE-V1",
        "phase": "evaluate",
        "scene": scene,
        "implementation_source_revision": implementation_revision,
        "evaluation_revision": evaluation_revision,
        "protocol": str(PROTOCOL_PATH.relative_to(REPOSITORY)),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "constants_sha256": sha256_file(CONSTANTS_PATH),
        "candidate_freeze": {
            "path": str(freeze_file),
            "sha256": sha256_file(freeze_file),
            "prepare_root": str(prepare_root),
        },
        "dataset": candidate_manifest["dataset"],
        "checkpoint": expected_checkpoint,
        "runtime": runtime,
        "extension": extension,
        "cameras": camera_rows,
        "resources": resource_record,
        "artifacts": artifact_records(
            {"evaluation_state": state_path, "result": result_path}
        ),
    }
    hashed_paths = [state_path, result_path]
    predicted = _predicted_artifact_bytes(output, manifest, hashed_paths)
    resource_record["artifact_bytes"] = predicted
    limit = int(PROTOCOL_CONSTANTS["resource_limits"]["artifact_bytes_per_scene"])
    if predicted > limit:
        raise ResourceLimitError(f"persistent artifacts need {predicted} bytes, limit is {limit}")
    manifest_path = write_json_new(output / "manifest.json", manifest)
    checksum_path = write_sha256s(output / "SHA256SUMS", [*hashed_paths, manifest_path])
    if directory_bytes(output) + len("complete\n") != predicted:
        raise RuntimeError("predicted and actual artifact byte counts differ")
    done_path = write_text_new(output / "DONE", "complete\n")
    if directory_bytes(output) != predicted:
        raise RuntimeError("final artifact byte count differs from its manifest")
    return {
        "result": result,
        "paths": {
            "state": str(state_path),
            "result": str(result_path),
            "manifest": str(manifest_path),
            "checksums": str(checksum_path),
            "done": str(done_path),
        },
    }


def _failure_marker(output: Path, marker: str, error: BaseException) -> None:
    if not output.is_dir() or (output / "DONE").exists():
        return
    path = output / marker
    if not path.exists():
        write_text_new(path, f"{type(error).__name__}: {error}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=tuple(PROTOCOL_CONSTANTS["scene_names"]))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--freeze-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    try:
        record = run_scene(
            scene=args.scene,
            dataset_root=args.dataset_root,
            model_root=args.model_root,
            prepare_root=args.prepare_root,
            freeze_file=args.freeze_file,
            output=args.output,
            physical_gpu=args.physical_gpu,
        )
    except (
        IdentityDriftError,
        SceneInvalidError,
        ResourceLimitError,
        ResourceMonitorError,
        OverflowError,
        ValueError,
    ) as error:
        _failure_marker(args.output.resolve(), "INVALID", error)
        raise
    except Exception as error:
        _failure_marker(args.output.resolve(), "FAILED", error)
        raise
    print(json.dumps(record["result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
