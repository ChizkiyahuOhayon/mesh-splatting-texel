"""Target-free checkpoint topology census for GoRFE-V1."""

import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch

from gorfe_v1_io import source_revision, write_json_new
from gorfe_v1_prepare_core import load_validated_checkpoint


SCHEMA = "gorfe-v1-target-free-topology-census-v1"


def census_faces(faces, *, vertex_count, example_limit=20, scan_chunk=1 << 22):
    if not torch.is_tensor(faces) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be a [F, 3] tensor")
    if faces.dtype not in (torch.int32, torch.int64) or faces.device.type != "cpu":
        raise TypeError("faces must be a CPU int32 or int64 tensor")
    if not isinstance(vertex_count, int) or vertex_count < 1:
        raise ValueError("vertex_count must be a positive integer")
    if not isinstance(example_limit, int) or example_limit < 0:
        raise ValueError("example_limit must be a nonnegative integer")
    if not isinstance(scan_chunk, int) or scan_chunk < 1:
        raise ValueError("scan_chunk must be a positive integer")
    if faces.numel() == 0:
        raise ValueError("checkpoint mesh must contain at least one face")

    faces = faces.to(torch.int64).contiguous()
    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    if bool(repeated.any()):
        raise ValueError("census requires faces with three distinct vertices")
    if bool((faces < 0).any()) or bool((faces >= vertex_count).any()):
        raise ValueError("face vertex index is outside the checkpoint vertex range")

    face_count = int(faces.shape[0])
    vertex_stride = int(faces.max()) + 1
    if vertex_stride > torch.iinfo(torch.int64).max // vertex_stride:
        raise OverflowError("packed edge key would overflow int64")

    keys = torch.empty((face_count, 3), dtype=torch.int64)
    a, b, c = faces.unbind(dim=1)
    keys[:, 0] = torch.minimum(a, b) * vertex_stride + torch.maximum(a, b)
    keys[:, 1] = torch.minimum(b, c) * vertex_stride + torch.maximum(b, c)
    keys[:, 2] = torch.minimum(c, a) * vertex_stride + torch.maximum(c, a)

    flat_keys = keys.reshape(-1)
    unique_keys, counts = torch.unique(
        flat_keys, sorted=True, return_counts=True
    )
    incidence_values, incidence_edge_counts = torch.unique(
        counts, sorted=True, return_counts=True
    )
    bad_mask = counts > 2
    bad_keys = unique_keys[bad_mask]
    bad_counts = counts[bad_mask]

    position_parts = []
    bad_row_parts = []
    if bad_keys.numel():
        for start in range(0, flat_keys.numel(), scan_chunk):
            stop = min(start + scan_chunk, flat_keys.numel())
            chunk = flat_keys[start:stop]
            locations = torch.searchsorted(bad_keys, chunk)
            in_range = locations < bad_keys.numel()
            safe = locations.clamp(max=int(bad_keys.numel()) - 1)
            matched = in_range & (bad_keys[safe] == chunk)
            if bool(matched.any()):
                local = torch.nonzero(matched, as_tuple=False).flatten()
                position_parts.append(local + start)
                bad_row_parts.append(locations[local])

        positions = torch.cat(position_parts)
        bad_rows = torch.cat(bad_row_parts)
        order = torch.argsort(bad_rows, stable=True)
        positions, bad_rows = positions[order], bad_rows[order]
        if int(positions.numel()) != int(bad_counts.sum()):
            raise RuntimeError("non-manifold occurrence census is incomplete")
    else:
        positions = torch.empty(0, dtype=torch.int64)
        bad_rows = torch.empty(0, dtype=torch.int64)

    examples = []
    duplicate_edges = 0
    distinct_edges = 0
    duplicate_face_occurrences = 0
    offset = 0
    for bad_row, incidence in enumerate(bad_counts.tolist()):
        stop = offset + int(incidence)
        if not bool((bad_rows[offset:stop] == bad_row).all()):
            raise RuntimeError("non-manifold occurrence grouping changed")
        incident_positions = positions[offset:stop]
        face_ids = torch.div(incident_positions, 3, rounding_mode="floor")
        local_slots = torch.remainder(incident_positions, 3)
        incident_faces = faces[face_ids]
        canonical_faces = torch.sort(incident_faces, dim=1).values
        distinct_faces = torch.unique(canonical_faces, dim=0)
        duplicate_occurrences = int(incident_faces.shape[0]) - int(
            distinct_faces.shape[0]
        )
        duplicate_edges += int(duplicate_occurrences > 0)
        distinct_edges += int(duplicate_occurrences == 0)
        duplicate_face_occurrences += duplicate_occurrences

        if len(examples) < example_limit:
            key = int(bad_keys[bad_row])
            examples.append(
                {
                    "edge": [key // vertex_stride, key % vertex_stride],
                    "incidence": int(incidence),
                    "face_ids": face_ids.tolist(),
                    "local_slots": local_slots.tolist(),
                    "faces": incident_faces.tolist(),
                    "canonical_faces": canonical_faces.tolist(),
                    "duplicate_incident_face_occurrences": duplicate_occurrences,
                }
            )
        offset = stop

    edge_count = int(unique_keys.numel())
    nonmanifold_count = int(bad_keys.numel())
    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "min_vertex_index": int(faces.min()),
        "max_vertex_index": int(faces.max()),
        "vertex_stride": vertex_stride,
        "repeated_vertex_faces": int(repeated.sum()),
        "faces_containing_vertex_zero": int((faces == 0).any(dim=1).sum()),
        "canonical_edge_count_including_nonmanifold": edge_count,
        "admissible_edge_count_if_nonmanifold_excluded": (
            edge_count - nonmanifold_count
        ),
        "incidence_histogram": {
            str(int(incidence)): int(edge_total)
            for incidence, edge_total in zip(
                incidence_values, incidence_edge_counts
            )
        },
        "nonmanifold_edge_count": nonmanifold_count,
        "nonmanifold_edge_fraction": nonmanifold_count / edge_count,
        "nonmanifold_face_local_occurrences": int(bad_counts.sum()),
        "maximum_edge_incidence": int(counts.max()),
        "nonmanifold_edges_with_duplicate_incident_face": duplicate_edges,
        "nonmanifold_edges_with_all_incident_faces_distinct": distinct_edges,
        "duplicate_incident_face_occurrences": duplicate_face_occurrences,
        "first_nonmanifold_edges": examples,
    }


def census_checkpoint(scene, model_root, expected_sha256):
    started = time.monotonic()
    state, inspection = load_validated_checkpoint(
        model_root,
        iteration=30000,
        expected_sha256=expected_sha256,
    )
    result = census_faces(
        state["_triangle_indices"], vertex_count=inspection.vertex_count
    )
    result.update(
        {
            "scene": scene,
            "checkpoint": {
                "path": inspection.path,
                "sha256": inspection.sha256,
                "bytes": inspection.bytes,
                "iteration": 30000,
            },
            "wall_seconds": time.monotonic() - started,
        }
    )
    del state
    gc.collect()
    return result


def run(args):
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    return {
        "schema": SCHEMA,
        "source_revision": source_revision(),
        "target_access": {
            "camera_metadata_read": False,
            "image_paths_opened": False,
            "rgb_decoded": False,
            "residual_or_loss_read": False,
        },
        "scenes": {
            "garden": census_checkpoint(
                "garden", args.garden_model, args.garden_sha256
            ),
            "room": census_checkpoint(
                "room", args.room_model, args.room_sha256
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--garden-model", type=Path, required=True)
    parser.add_argument("--garden-sha256", required=True)
    parser.add_argument("--room-model", type=Path, required=True)
    parser.add_argument("--room-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    write_json_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
