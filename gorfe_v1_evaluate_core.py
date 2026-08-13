"""CPU-only integration from frozen GoRFE-V1 support to scene metrics."""

from dataclasses import dataclass

import torch

from gorfe_v1 import (
    FAMILY_NAMES,
    FOLD_COUNT,
    PROTOCOL_CONSTANTS,
    TYPE_COSTS,
    decide_scene_family,
    evaluate_family,
    nested_scores,
)
from gorfe_v1_stream import CarrierStatistics, StreamStatistics


MINIMUM_FOLD_SUPPORT_PIXELS = int(PROTOCOL_CONSTANTS["minimum_fold_support_pixels"])
MINIMUM_FOLD_SUPPORT_CAMERAS = int(PROTOCOL_CONSTANTS["minimum_fold_support_cameras"])
SH1_RANK_RELATIVE_TRACE_THRESHOLD = float(
    PROTOCOL_CONSTANTS["sh1_rank_relative_trace_threshold"]
)


class SceneInvalidError(RuntimeError):
    """A sealed V1 scene cannot produce a scientific result."""


class FreezeMismatchError(SceneInvalidError):
    """Evaluation design identity differs from the target-free freeze."""


@dataclass(frozen=True)
class EligibilityMasks:
    dc: torch.Tensor
    sh1: torch.Tensor


@dataclass(frozen=True)
class PolicyScores:
    primary: torch.Tensor
    outcome: torch.Tensor
    raw_residual: torch.Tensor
    same_view_gain: torch.Tensor
    rhs_norm: torch.Tensor
    coverage: torch.Tensor


@dataclass(frozen=True)
class SceneEvaluation:
    type_ids: torch.Tensor
    endpoints: torch.Tensor
    result: dict


def _validate_carrier(carrier, feature_dim, name):
    gram = carrier.gram
    if not torch.is_tensor(gram) or gram.ndim != 4:
        raise SceneInvalidError(f"{name}.gram must have shape [groups, 4, Q, Q]")
    groups = gram.shape[0]
    expected = {
        "gram": (groups, FOLD_COUNT, feature_dim, feature_dim),
        "rhs": (groups, FOLD_COUNT, feature_dim, 3),
        "support_rss": (groups, FOLD_COUNT),
        "support_pixels": (groups, FOLD_COUNT),
        "support_cameras": (groups, FOLD_COUNT),
    }
    for field, shape in expected.items():
        value = getattr(carrier, field)
        if not torch.is_tensor(value) or value.shape != shape:
            raise SceneInvalidError(f"{name}.{field} must have shape {list(shape)}")
        if value.device.type != "cpu":
            raise SceneInvalidError(f"{name}.{field} must be a CPU tensor")
    for field in ("gram", "rhs", "support_rss"):
        if getattr(carrier, field).dtype != torch.float64:
            raise SceneInvalidError(f"{name}.{field} must use float64")
    for field in ("support_pixels", "support_cameras"):
        if getattr(carrier, field).dtype != torch.int64:
            raise SceneInvalidError(f"{name}.{field} must use int64")
    finite_gram = torch.isfinite(gram).all(dim=-1).all(dim=-1)
    symmetric_gram = (gram == gram.transpose(-1, -2)).all(dim=-1).all(dim=-1)
    if bool((finite_gram & ~symmetric_gram).any()):
        raise SceneInvalidError(f"{name}.gram is not exactly symmetric")
    if bool((carrier.support_pixels < 0).any()) or bool((carrier.support_cameras < 0).any()):
        raise SceneInvalidError(f"{name} support counts must be nonnegative")
    if bool((carrier.support_cameras > carrier.support_pixels).any()):
        raise SceneInvalidError(f"{name} camera support cannot exceed pixel support")
    return groups


def _validate_stream(statistics, name):
    dc_groups = _validate_carrier(statistics.dc, 1, f"{name}.dc")
    sh1_groups = _validate_carrier(statistics.sh1, 3, f"{name}.sh1")
    if dc_groups != sh1_groups:
        raise SceneInvalidError(f"{name} DC and SH1 candidate counts differ")
    full_rss = statistics.fold_full_rss
    if (
        not torch.is_tensor(full_rss)
        or full_rss.shape != (FOLD_COUNT,)
        or full_rss.dtype != torch.float64
        or full_rss.device.type != "cpu"
    ):
        raise SceneInvalidError(f"{name}.fold_full_rss must be CPU float64 [4]")
    return dc_groups


def _target_free_is_clean(statistics):
    for name, carrier in (("dc", statistics.dc), ("sh1", statistics.sh1)):
        if not bool(torch.isfinite(carrier.rhs).all()) or bool((carrier.rhs != 0).any()):
            raise SceneInvalidError(f"target-free {name} rhs is not identically zero")
        if not bool(torch.isfinite(carrier.support_rss).all()) or bool(
            (carrier.support_rss != 0).any()
        ):
            raise SceneInvalidError(f"target-free {name} RSS is not identically zero")
    if not bool(torch.isfinite(statistics.fold_full_rss).all()) or bool(
        (statistics.fold_full_rss != 0).any()
    ):
        raise SceneInvalidError("target-free full-fold RSS is not identically zero")


def _base_eligible(carrier):
    finite = torch.isfinite(carrier.gram).all(dim=-1).all(dim=-1)
    trace = carrier.gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    support = (
        (carrier.support_pixels >= MINIMUM_FOLD_SUPPORT_PIXELS)
        & (carrier.support_cameras >= MINIMUM_FOLD_SUPPORT_CAMERAS)
    )
    return finite & torch.isfinite(trace) & (trace > 0) & support, trace


def derive_eligibility(statistics):
    """Derive the only masks allowed to be sealed by target-free preparation."""
    _validate_stream(statistics, "target_free")
    _target_free_is_clean(statistics)
    dc_per_fold, _ = _base_eligible(statistics.dc)
    sh1_per_fold, sh1_trace = _base_eligible(statistics.sh1)

    safe = torch.where(
        sh1_per_fold[..., None, None],
        statistics.sh1.gram,
        torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3),
    )
    smallest_eigenvalue = torch.linalg.eigvalsh(safe)[..., 0]
    full_rank = smallest_eigenvalue > (
        SH1_RANK_RELATIVE_TRACE_THRESHOLD * sh1_trace / 3.0
    )
    return EligibilityMasks(
        dc=dc_per_fold.all(dim=1),
        sh1=(sh1_per_fold & full_rank).all(dim=1),
    )


def _validate_masks(masks, groups):
    for name in ("dc", "sh1"):
        value = getattr(masks, name)
        if (
            not torch.is_tensor(value)
            or value.shape != (groups,)
            or value.dtype != torch.bool
            or value.device.type != "cpu"
        ):
            raise FreezeMismatchError(f"frozen {name} mask must be CPU bool [{groups}]")


def _float64_bitwise_equal(first, second):
    return (
        first.shape == second.shape
        and first.dtype == second.dtype == torch.float64
        and first.device.type == second.device.type == "cpu"
        and torch.equal(
            first.contiguous().view(torch.int64), second.contiguous().view(torch.int64)
        )
    )


def validate_evaluation_identity(target_free, evaluation, frozen_masks):
    """Refuse any design or eligibility drift before target statistics are used."""
    groups = _validate_stream(target_free, "target_free")
    _target_free_is_clean(target_free)
    if _validate_stream(evaluation, "evaluation") != groups:
        raise FreezeMismatchError("evaluation candidate count differs from the freeze")
    _validate_masks(frozen_masks, groups)
    recomputed = derive_eligibility(target_free)
    if not torch.equal(frozen_masks.dc, recomputed.dc):
        raise FreezeMismatchError("frozen DC eligibility mask does not match support")
    if not torch.equal(frozen_masks.sh1, recomputed.sh1):
        raise FreezeMismatchError("frozen SH1 eligibility mask does not match support")

    for name in ("dc", "sh1"):
        frozen = getattr(target_free, name)
        observed = getattr(evaluation, name)
        if not _float64_bitwise_equal(frozen.gram, observed.gram):
            raise FreezeMismatchError(f"evaluation {name} Gram differs bitwise from freeze")
        if not torch.equal(frozen.support_pixels, observed.support_pixels):
            raise FreezeMismatchError(f"evaluation {name} pixel support differs from freeze")
        if not torch.equal(frozen.support_cameras, observed.support_cameras):
            raise FreezeMismatchError(f"evaluation {name} camera support differs from freeze")

        eligible = getattr(frozen_masks, name)
        for field in ("gram", "rhs", "support_rss"):
            if not bool(torch.isfinite(getattr(observed, field)[eligible]).all()):
                raise SceneInvalidError(f"sealed eligible {name} {field} is nonfinite")
        if bool((observed.support_rss[eligible] < 0).any()):
            raise SceneInvalidError(f"sealed eligible {name} RSS is negative")

    if not bool(torch.isfinite(evaluation.fold_full_rss).all()) or bool(
        (evaluation.fold_full_rss <= 0).any()
    ):
        raise SceneInvalidError("evaluation full-fold RSS must be finite and positive")


def _validate_endpoints(endpoints, groups):
    if (
        not torch.is_tensor(endpoints)
        or endpoints.shape != (groups, 2)
        or endpoints.dtype != torch.int64
        or endpoints.device.type != "cpu"
    ):
        raise SceneInvalidError(f"candidate_endpoints must be CPU int64 [{groups}, 2]")
    if bool((endpoints[:, 0] < 0).any()) or bool((endpoints[:, 0] >= endpoints[:, 1]).any()):
        raise SceneInvalidError("candidate endpoints must be canonical nonnegative pairs")
    if torch.unique(endpoints, dim=0).shape[0] != groups:
        raise SceneInvalidError("candidate endpoints must be unique")


def _subset(carrier, mask):
    return CarrierStatistics(
        gram=carrier.gram[mask],
        rhs=carrier.rhs[mask],
        support_rss=carrier.support_rss[mask],
        support_pixels=carrier.support_pixels[mask],
        support_cameras=carrier.support_cameras[mask],
    )


def _combine_policy_scores(dc, sh1):
    fields = {}
    for name in (
        "primary",
        "outcome",
        "raw_residual",
        "same_view_gain",
        "rhs_norm",
        "coverage",
    ):
        fields[name] = torch.cat((getattr(dc, name), getattr(sh1, name)), dim=0)
    return PolicyScores(**fields)


def _nested_or_empty(carrier, mask):
    if bool(mask.any()):
        return nested_scores(_subset(carrier, mask))
    empty = torch.empty((0, FOLD_COUNT), dtype=torch.float64)
    return PolicyScores(
        primary=empty.clone(),
        outcome=empty.clone(),
        raw_residual=empty.clone(),
        same_view_gain=empty.clone(),
        rhs_norm=empty.clone(),
        coverage=empty.clone(),
    )


def evaluate_scene(
    *,
    scene,
    candidate_endpoints,
    target_free_statistics,
    evaluation_statistics,
    frozen_masks,
):
    """Validate one frozen scene and produce the three preregistered family metrics."""
    if scene not in PROTOCOL_CONSTANTS["scene_names"]:
        raise SceneInvalidError(f"unknown locked scene {scene!r}")
    groups = _validate_stream(target_free_statistics, "target_free")
    _validate_endpoints(candidate_endpoints, groups)
    validate_evaluation_identity(
        target_free_statistics, evaluation_statistics, frozen_masks
    )

    try:
        dc_nested = _nested_or_empty(evaluation_statistics.dc, frozen_masks.dc)
        sh1_nested = _nested_or_empty(evaluation_statistics.sh1, frozen_masks.sh1)
    except (ValueError, RuntimeError) as error:
        raise SceneInvalidError(
            "a sealed eligible group failed nested GCV; no group was dropped"
        ) from error

    dc_rows = torch.nonzero(frozen_masks.dc, as_tuple=False).flatten()
    sh1_rows = torch.nonzero(frozen_masks.sh1, as_tuple=False).flatten()
    type_ids = torch.cat(
        (
            torch.zeros(dc_rows.numel(), dtype=torch.int64),
            torch.ones(sh1_rows.numel(), dtype=torch.int64),
        )
    )
    endpoints = torch.cat(
        (candidate_endpoints[dc_rows], candidate_endpoints[sh1_rows]), dim=0
    )
    combined = _combine_policy_scores(dc_nested, sh1_nested)

    families = {}
    try:
        for family in FAMILY_NAMES:
            metrics = evaluate_family(
                combined,
                scene,
                type_ids,
                endpoints,
                evaluation_statistics.fold_full_rss,
                family,
            )
            families[family] = {
                "metrics": metrics,
                "decision": decide_scene_family(metrics),
            }
    except (ValueError, RuntimeError) as error:
        raise SceneInvalidError(
            "a sealed eligible group failed family evaluation; no group was dropped"
        ) from error

    dc_count = int(frozen_masks.dc.sum())
    sh1_count = int(frozen_masks.sh1.sum())
    costs = TYPE_COSTS.tolist()
    result = {
        "scene": scene,
        "candidate_edges": groups,
        "eligible_groups": {"DC": dc_count, "SH1": sh1_count},
        "eligible_cost_units": {
            "DC": dc_count * costs[0],
            "SH1": sh1_count * costs[1],
            "MIXED": dc_count * costs[0] + sh1_count * costs[1],
        },
        "passing_families": [
            family for family in FAMILY_NAMES if families[family]["decision"]["pass"]
        ],
        "families": families,
    }
    return SceneEvaluation(type_ids=type_ids, endpoints=endpoints, result=result)
