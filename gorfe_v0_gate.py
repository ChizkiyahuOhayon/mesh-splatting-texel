"""Synthetic dense-oracle gate for GoRFE's sparse fold statistics."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from edgeval_core import deterministic_camera_folds
from gorfe_v0 import (
    CameraDesignRows,
    dense_fold_statistics,
    heldout_signed_gain,
    streaming_fold_statistics,
)


SEED = 20260811
FOLD_COUNT = 4
GROUP_COUNT = 3
ATOL = 1e-11
RTOL = 1e-11
MEMORY_CEILING_BYTES = 4 * 1024 * 1024
WALL_TIME_CEILING_SECONDS = 10.0


_FRAGMENT_SPECS = (
    # pixel, group, face, depth, P2 edge basis, alpha, transmittance
    (0, 0, 0, 1.25, 0.72, 0.35, 1.00),
    (4, 1, 2, 1.60, 0.54, 0.42, 0.91),
    (8, 2, 4, 2.10, 0.61, 0.31, 0.76),
    (2, 0, 0, 1.35, 0.43, 0.51, 0.88),
    (6, 1, 2, 1.80, 0.66, 0.29, 0.81),
    (0, 0, 1, 2.50, 0.72, 0.28, 0.65),
    (10, 2, 4, 2.35, 0.48, 0.47, 0.69),
    (6, 1, 3, 2.75, 0.66, 0.22, 0.58),
    (2, 2, 5, 3.10, 0.37, 0.39, 0.52),
)


def build_fixture(feature_dim):
    if feature_dim not in (1, 3):
        raise ValueError("the locked fixture only defines Q=1 and Q=3")
    generator = torch.Generator().manual_seed(SEED)
    names = [f"view_{index:02d}" for index in range(8)]
    folds = deterministic_camera_folds(names, FOLD_COUNT)
    cameras = []
    metadata = {}
    for camera_index, name in enumerate(names):
        residual_pixel_ids = torch.tensor([0, 2, 4, 6, 8, 10], dtype=torch.int64)
        residuals = 0.25 * torch.randn((6, 3), generator=generator, dtype=torch.float64)
        residuals += torch.tensor(
            [0.03 * camera_index, -0.02 * camera_index, 0.01 * camera_index],
            dtype=torch.float64,
        )
        features = []
        faces = []
        depths = []
        for pixel, _group, face, depth, edge_value, alpha, transmittance in _FRAGMENT_SPECS:
            scalar = edge_value * alpha * transmittance * (1.0 + 0.015 * camera_index)
            angular = 0.18 * torch.randn(3, generator=generator, dtype=torch.float64)
            angular += torch.tensor(
                [
                    -0.31 + 0.025 * camera_index + 0.012 * face,
                    0.14 + 0.017 * pixel - 0.009 * camera_index,
                    0.27 - 0.011 * pixel + 0.014 * face,
                ],
                dtype=torch.float64,
            )
            if feature_dim == 1:
                features.append(torch.tensor([scalar], dtype=torch.float64))
            else:
                features.append(scalar * angular)
            faces.append(face)
            depths.append(depth)
        cameras.append(
            CameraDesignRows(
                name=name,
                fold=folds[name],
                residual_pixel_ids=residual_pixel_ids,
                residuals=residuals,
                pixel_ids=torch.tensor(
                    [spec[0] for spec in _FRAGMENT_SPECS], dtype=torch.int64
                ),
                group_ids=torch.tensor(
                    [spec[1] for spec in _FRAGMENT_SPECS], dtype=torch.int64
                ),
                features=torch.stack(features),
            )
        )
        metadata[name] = {
            "face_ids": torch.tensor(faces, dtype=torch.int64),
            "depths": torch.tensor(depths, dtype=torch.float64),
        }
    return cameras, metadata


def _reordered(cameras, mode):
    generator = torch.Generator().manual_seed(SEED + 1)
    output = []
    camera_order = list(range(len(cameras)))
    if mode == "reverse":
        camera_order.reverse()
    elif mode == "permuted":
        camera_order = torch.randperm(len(cameras), generator=generator).tolist()
    elif mode != "original":
        raise ValueError(f"unknown order mode {mode!r}")
    for camera_index in camera_order:
        camera = cameras[camera_index]
        if mode == "reverse":
            contribution_order = torch.arange(camera.pixel_ids.numel() - 1, -1, -1)
            residual_order = torch.arange(camera.residual_pixel_ids.numel() - 1, -1, -1)
        elif mode == "permuted":
            contribution_order = torch.randperm(camera.pixel_ids.numel(), generator=generator)
            residual_order = torch.randperm(camera.residual_pixel_ids.numel(), generator=generator)
        else:
            contribution_order = torch.arange(camera.pixel_ids.numel())
            residual_order = torch.arange(camera.residual_pixel_ids.numel())
        output.append(
            CameraDesignRows(
                name=camera.name,
                fold=camera.fold,
                residual_pixel_ids=camera.residual_pixel_ids[residual_order],
                residuals=camera.residuals[residual_order],
                pixel_ids=camera.pixel_ids[contribution_order],
                group_ids=camera.group_ids[contribution_order],
                features=camera.features[contribution_order],
            )
        )
    return output


def _maximum_absolute_error(first, second):
    return float((first - second).abs().max()) if first.numel() else 0.0


def _close(first, second):
    return bool(torch.allclose(first, second, atol=ATOL, rtol=RTOL))


def _duplicate_cross_term_check(camera, metadata, feature_dim):
    duplicate = (camera.pixel_ids == 0) & (camera.group_ids == 0)
    rows = torch.nonzero(duplicate, as_tuple=False).flatten()
    faces = metadata["face_ids"][rows]
    depths = metadata["depths"][rows]
    mini = CameraDesignRows(
        name="shared_edge_probe",
        fold=0,
        residual_pixel_ids=torch.tensor([0], dtype=torch.int64),
        residuals=camera.residuals[:1],
        pixel_ids=camera.pixel_ids[rows],
        group_ids=torch.zeros(rows.numel(), dtype=torch.int64),
        features=camera.features[rows],
    )
    statistics, _ = streaming_fold_statistics(
        [mini], 1, FOLD_COUNT, feature_dim, chunk_size=1
    )
    first, second = mini.features
    naive = torch.outer(first, first) + torch.outer(second, second)
    cross = torch.outer(first, second) + torch.outer(second, first)
    observed = statistics.gram[0, 0]
    return {
        "two_incident_faces_same_pixel_group": bool(
            rows.numel() == 2
            and set(faces.tolist()) == {0, 1}
            and set(depths.tolist()) == {1.25, 2.5}
        ),
        "cross_term_norm": float(torch.linalg.vector_norm(cross)),
        "naive_gram_error": _maximum_absolute_error(observed, naive),
        "cross_term_is_present": _close(observed - naive, cross),
    }


def _run_group(feature_dim):
    cameras, metadata = build_fixture(feature_dim)
    dense = dense_fold_statistics(cameras, GROUP_COUNT, FOLD_COUNT, feature_dim)
    reference, diagnostics = streaming_fold_statistics(
        cameras, GROUP_COUNT, FOLD_COUNT, feature_dim
    )
    dense_fit = heldout_signed_gain(dense)
    reference_fit = heldout_signed_gain(reference)

    invariant = True
    maximum_order_error = 0.0
    for mode in ("reverse", "permuted"):
        ordered = _reordered(cameras, mode)
        for chunk_size in (1, 2, 7, None):
            candidate, _ = streaming_fold_statistics(
                ordered,
                GROUP_COUNT,
                FOLD_COUNT,
                feature_dim,
                chunk_size=chunk_size,
            )
            candidate_fit = heldout_signed_gain(candidate)
            comparisons = (
                (candidate.gram, reference.gram),
                (candidate.rhs, reference.rhs),
                (candidate.support_rss, reference.support_rss),
                (candidate_fit.signed_gain, reference_fit.signed_gain),
            )
            invariant = invariant and all(_close(first, second) for first, second in comparisons)
            invariant = invariant and torch.equal(
                candidate.support_pixels, reference.support_pixels
            )
            maximum_order_error = max(
                maximum_order_error,
                *(_maximum_absolute_error(first, second) for first, second in comparisons),
            )

    duplicate = _duplicate_cross_term_check(cameras[0], metadata[cameras[0].name], feature_dim)
    declared_shapes = (
        reference.gram.shape == (GROUP_COUNT, FOLD_COUNT, feature_dim, feature_dim)
        and reference.rhs.shape == (GROUP_COUNT, FOLD_COUNT, feature_dim, 3)
        and reference.support_rss.shape == (GROUP_COUNT, FOLD_COUNT)
        and reference.support_pixels.shape == (GROUP_COUNT, FOLD_COUNT)
    )
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (
            reference.gram,
            reference.rhs,
            reference.support_rss,
            reference_fit.ridge,
            reference_fit.coefficients,
            reference_fit.signed_gain,
        )
    )
    float64 = all(
        value.dtype == torch.float64
        for value in (reference.gram, reference.rhs, reference.support_rss)
    )
    checks = {
        "gram_matches_dense": _close(reference.gram, dense.gram),
        "rhs_matches_dense": _close(reference.rhs, dense.rhs),
        "support_rss_matches_dense": _close(reference.support_rss, dense.support_rss),
        "support_pixels_match_dense": torch.equal(
            reference.support_pixels, dense.support_pixels
        ),
        "heldout_signed_gain_matches_dense": _close(
            reference_fit.signed_gain, dense_fit.signed_gain
        ),
        "order_and_chunk_invariant": invariant,
        "two_incident_faces_same_pixel_group": duplicate[
            "two_incident_faces_same_pixel_group"
        ],
        "duplicate_cross_term_is_nonzero": duplicate["cross_term_norm"] > 0.0,
        "duplicate_cross_term_is_present": duplicate["cross_term_is_present"],
        "naive_per_fragment_gram_fails": duplicate["naive_gram_error"] > 1e-8,
        "outputs_are_finite": finite,
        "outputs_have_declared_shapes_and_dtypes": declared_shapes and float64,
        "temporary_memory_within_ceiling": (
            diagnostics.estimated_peak_temporary_bytes <= MEMORY_CEILING_BYTES
        ),
    }
    return {
        "feature_dim": feature_dim,
        "checks": checks,
        "diagnostics": asdict(diagnostics),
        "maximum_errors": {
            "gram_dense": _maximum_absolute_error(reference.gram, dense.gram),
            "rhs_dense": _maximum_absolute_error(reference.rhs, dense.rhs),
            "support_rss_dense": _maximum_absolute_error(
                reference.support_rss, dense.support_rss
            ),
            "heldout_signed_gain_dense": _maximum_absolute_error(
                reference_fit.signed_gain, dense_fit.signed_gain
            ),
            "order_or_chunk": maximum_order_error,
        },
        "duplicate_probe": duplicate,
        "heldout_signed_gain": reference_fit.signed_gain.tolist(),
        "support_pixels": reference.support_pixels.tolist(),
    }


def run():
    started = time.perf_counter()
    groups = {
        "p2_dc": _run_group(1),
        "p2_sh1": _run_group(3),
    }
    wall_time = time.perf_counter() - started
    checks = {
        f"{group}_{name}": value
        for group, result in groups.items()
        for name, value in result["checks"].items()
    }
    checks["wall_time_within_ceiling"] = wall_time <= WALL_TIME_CEILING_SECONDS
    result = {
        "experiment": "GoRFE-V0",
        "fixture": "synthetic_multi_triangle_multi_depth",
        "seed": SEED,
        "torch": torch.__version__,
        "device": "cpu",
        "float_dtype": "float64",
        "absolute_tolerance": ATOL,
        "relative_tolerance": RTOL,
        "wall_time_seconds": wall_time,
        "checks": checks,
        "groups": groups,
    }
    result["decision"] = "pass" if all(checks.values()) else "fail"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if result["decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
