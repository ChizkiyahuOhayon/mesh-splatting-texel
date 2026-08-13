"""Typed serialization boundary for GoRFE-V1 candidate and fold state."""

from dataclasses import fields

import torch

from gorfe_v1_evaluate_core import EligibilityMasks
from gorfe_v1_stream import CarrierStatistics, StreamStatistics


SCHEMA = "gorfe-v1-candidate-state-v2"
CANDIDATE_MANIFEST_SCHEMA = "gorfe-v1-candidate-manifest-v2"


def _carrier_to_payload(carrier):
    return {field.name: getattr(carrier, field.name).detach().cpu() for field in fields(carrier)}


def _carrier_from_payload(value, name):
    expected = {field.name for field in fields(CarrierStatistics)}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} carrier fields differ from {sorted(expected)}")
    if any(not torch.is_tensor(value[key]) for key in expected):
        raise TypeError(f"{name} carrier values must be tensors")
    return CarrierStatistics(**{key: value[key] for key in expected})


def stream_to_payload(statistics):
    return {
        "dc": _carrier_to_payload(statistics.dc),
        "sh1": _carrier_to_payload(statistics.sh1),
        "fold_full_rss": statistics.fold_full_rss.detach().cpu(),
    }


def stream_from_payload(value):
    if not isinstance(value, dict) or set(value) != {"dc", "sh1", "fold_full_rss"}:
        raise ValueError("stream payload has unexpected fields")
    if not torch.is_tensor(value["fold_full_rss"]):
        raise TypeError("fold_full_rss must be a tensor")
    return StreamStatistics(
        dc=_carrier_from_payload(value["dc"], "dc"),
        sh1=_carrier_from_payload(value["sh1"], "sh1"),
        fold_full_rss=value["fold_full_rss"],
    )


def candidate_state_payload(*, scene, topology, statistics, eligibility):
    return {
        "schema": SCHEMA,
        "scene": scene,
        "edge_count": int(topology.edge_count),
        "vertex_stride": int(topology.vertex_stride),
        "candidate_edge_indices": topology.candidate_edge_indices.detach().cpu(),
        "candidate_edges": topology.candidate_edges.detach().cpu(),
        "candidate_hashes": torch.from_numpy(topology.candidate_hashes.copy()),
        "candidate_incident_face_counts": topology.candidate_incident_face_counts.detach().cpu(),
        "face_candidate_edges": topology.face_candidate_edges.detach().cpu(),
        "target_free_statistics": stream_to_payload(statistics),
        "eligible_dc": eligibility.dc.detach().cpu(),
        "eligible_sh1": eligibility.sh1.detach().cpu(),
    }


def load_candidate_state(path, *, expected_scene, expected_face_count=None):
    value = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema",
        "scene",
        "edge_count",
        "vertex_stride",
        "candidate_edge_indices",
        "candidate_edges",
        "candidate_hashes",
        "candidate_incident_face_counts",
        "face_candidate_edges",
        "target_free_statistics",
        "eligible_dc",
        "eligible_sh1",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("candidate state has unexpected fields")
    if value["schema"] != SCHEMA or value["scene"] != expected_scene:
        raise ValueError("candidate state schema or scene identity changed")
    tensors = (
        "candidate_edge_indices",
        "candidate_edges",
        "candidate_hashes",
        "candidate_incident_face_counts",
        "face_candidate_edges",
        "eligible_dc",
        "eligible_sh1",
    )
    if any(not torch.is_tensor(value[name]) or value[name].device.type != "cpu" for name in tensors):
        raise TypeError("candidate state identity fields must be CPU tensors")
    edge_count = value["edge_count"]
    vertex_stride = value["vertex_stride"]
    if (
        not isinstance(edge_count, int)
        or isinstance(edge_count, bool)
        or edge_count < 0
        or not isinstance(vertex_stride, int)
        or isinstance(vertex_stride, bool)
        or vertex_stride < 0
    ):
        raise ValueError("edge_count and vertex_stride must be nonnegative integers")
    groups = value["candidate_edges"].shape[0]
    if value["candidate_edges"].shape != (groups, 2) or value["candidate_edges"].dtype != torch.int64:
        raise ValueError("candidate_edges must be CPU int64 [G,2]")
    if (
        value["candidate_edge_indices"].shape != (groups,)
        or value["candidate_edge_indices"].dtype != torch.int64
        or value["candidate_hashes"].shape != (groups,)
        or value["candidate_hashes"].dtype != torch.uint64
        or value["candidate_incident_face_counts"].shape != (groups,)
        or value["candidate_incident_face_counts"].dtype != torch.int32
    ):
        raise ValueError("candidate index/hash/incidence tensors have invalid contracts")
    if groups > edge_count:
        raise ValueError("candidate count exceeds canonical edge count")
    indices = value["candidate_edge_indices"]
    if bool((indices < 0).any()) or bool((indices >= edge_count).any()):
        raise ValueError("candidate canonical-edge index is out of range")
    if groups > 1 and not bool((indices[1:] > indices[:-1]).all()):
        raise ValueError("candidate canonical-edge indices must be strictly increasing")
    endpoints = value["candidate_edges"]
    if bool((endpoints[:, 0] < 0).any()) or bool((endpoints[:, 0] >= endpoints[:, 1]).any()):
        raise ValueError("candidate endpoints must be canonical nonnegative pairs")
    if groups and (
        vertex_stride <= int(endpoints.max())
        or (
            groups > 1
            and not bool(
                (
                    (endpoints[1:, 0] > endpoints[:-1, 0])
                    | (
                        (endpoints[1:, 0] == endpoints[:-1, 0])
                        & (endpoints[1:, 1] > endpoints[:-1, 1])
                    )
                ).all()
            )
        )
    ):
        raise ValueError("candidate endpoints or vertex_stride are inconsistent")
    incidences = value["candidate_incident_face_counts"]
    if bool((incidences < 1).any()):
        raise ValueError("candidate incidence counts must be positive")
    face_map = value["face_candidate_edges"]
    if face_map.ndim != 2 or face_map.shape[1] != 3:
        raise ValueError("face_candidate_edges must have shape [F,3]")
    if face_map.dtype != torch.int32:
        raise TypeError("face_candidate_edges must use int32")
    if expected_face_count is not None and (
        not isinstance(expected_face_count, int)
        or isinstance(expected_face_count, bool)
        or face_map.shape[0] != expected_face_count
    ):
        raise ValueError("candidate face map differs from checkpoint face count")
    if bool((face_map < -1).any()) or (groups == 0 and bool((face_map >= 0).any())):
        raise ValueError("candidate face map contains an invalid compact id")
    if groups and bool((face_map >= groups).any()):
        raise ValueError("candidate face map contains an out-of-range compact id")
    observed_incidence = torch.bincount(
        face_map[face_map >= 0].to(torch.int64), minlength=groups
    )
    if not torch.equal(observed_incidence, incidences.to(torch.int64)):
        raise ValueError("candidate face map does not reproduce incidence counts")
    for name in ("eligible_dc", "eligible_sh1"):
        if value[name].shape != (groups,) or value[name].dtype != torch.bool:
            raise ValueError(f"{name} must be bool [G]")
    statistics = stream_from_payload(value["target_free_statistics"])
    masks = EligibilityMasks(value["eligible_dc"], value["eligible_sh1"])
    return value, statistics, masks
