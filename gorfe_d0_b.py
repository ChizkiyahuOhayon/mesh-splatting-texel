"""Exact joint-utility arithmetic and frozen GoRFE-D0-B mechanism reading."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from gorfe_v1_stream import reduce_camera_design


_CONSTANTS_PATH = (
    Path(__file__).resolve().parent
    / "experiments"
    / "gorfe_d0_b"
    / "protocol_constants.json"
)
PROTOCOL_CONSTANTS = json.loads(_CONSTANTS_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class JointTerms:
    linear: float
    diagonal_quadratic: float
    joint_quadratic: float
    input_rows: int
    reduced_rows: int

    @property
    def additive_gain(self) -> float:
        return self.linear - self.diagonal_quadratic

    @property
    def joint_gain(self) -> float:
        return self.linear - self.joint_quadratic

    @property
    def interaction_penalty(self) -> float:
        return self.joint_quadratic - self.diagonal_quadratic

    def __add__(self, other: object) -> "JointTerms":
        if not isinstance(other, JointTerms):
            return NotImplemented
        return JointTerms(
            linear=self.linear + other.linear,
            diagonal_quadratic=self.diagonal_quadratic + other.diagonal_quadratic,
            joint_quadratic=self.joint_quadratic + other.joint_quadratic,
            input_rows=self.input_rows + other.input_rows,
            reduced_rows=self.reduced_rows + other.reduced_rows,
        )


def _require_coefficients(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if not torch.is_tensor(value) or value.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must use float64")
    if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite on CPU")


def camera_joint_terms(
    *,
    pixel_count: int,
    pixel_ids: torch.Tensor,
    group_ids: torch.Tensor,
    features: torch.Tensor,
    residuals: torch.Tensor,
    dc_coefficients: torch.Tensor,
    sh1_coefficients: torch.Tensor,
) -> JointTerms:
    """Evaluate one camera after exact ``(pixel, canonical edge)`` reduction.

    The coefficient banks are zero outside one frozen portfolio.  DC and SH1
    remain separate costed groups even when both occupy the same canonical edge.
    """

    if (
        not isinstance(pixel_count, int)
        or isinstance(pixel_count, bool)
        or pixel_count < 1
    ):
        raise ValueError("pixel_count must be a positive integer")
    if not torch.is_tensor(residuals) or residuals.shape != (pixel_count, 3):
        raise ValueError("residuals must have shape [pixel_count, 3]")
    if residuals.dtype != torch.float64:
        raise TypeError("residuals must use float64")
    if residuals.device.type != "cpu" or not bool(torch.isfinite(residuals).all()):
        raise ValueError("residuals must be finite on CPU")
    if not torch.is_tensor(dc_coefficients) or dc_coefficients.ndim != 3:
        raise ValueError("dc_coefficients must have shape [group_count, 1, 3]")
    group_count = int(dc_coefficients.shape[0])
    if group_count < 1:
        raise ValueError("coefficient banks must contain at least one candidate edge")
    _require_coefficients(dc_coefficients, (group_count, 1, 3), "dc_coefficients")
    _require_coefficients(sh1_coefficients, (group_count, 3, 3), "sh1_coefficients")
    for value, name in (
        (pixel_ids, "pixel_ids"),
        (group_ids, "group_ids"),
        (features, "features"),
    ):
        if not torch.is_tensor(value) or value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")

    reduced_pixels, reduced_groups, reduced = reduce_camera_design(
        pixel_ids, group_ids, features, group_count
    )
    if reduced_pixels.numel() and int(reduced_pixels.max()) >= pixel_count:
        raise ValueError("a design row references a pixel outside the camera")

    dc = torch.einsum(
        "mi,mic->mc", reduced[:, :1], dc_coefficients[reduced_groups]
    )
    sh1 = torch.einsum(
        "mi,mic->mc", reduced[:, 1:], sh1_coefficients[reduced_groups]
    )
    row_correction = dc + sh1
    correction = torch.zeros((pixel_count, 3), dtype=torch.float64)
    correction.index_add_(0, reduced_pixels, row_correction)

    linear = 2.0 * torch.sum(correction * residuals)
    diagonal = torch.sum(dc.square()) + torch.sum(sh1.square())
    joint = torch.sum(correction.square())
    values = (linear, diagonal, joint)
    if not all(bool(torch.isfinite(value)) for value in values):
        raise ValueError("joint-utility terms must be finite")
    tolerance = float(PROTOCOL_CONSTANTS["numeric_relative_tolerance"])
    if float(diagonal) < -tolerance or float(joint) < -tolerance:
        raise ValueError("squared correction norms must be nonnegative")
    return JointTerms(
        linear=float(linear),
        diagonal_quadratic=max(0.0, float(diagonal)),
        joint_quadratic=max(0.0, float(joint)),
        input_rows=int(pixel_ids.numel()),
        reduced_rows=int(reduced_pixels.numel()),
    )


def cv4(values: torch.Tensor) -> float:
    if not torch.is_tensor(values) or values.shape != (4,):
        raise ValueError("CV4 requires exactly four folds")
    if values.dtype != torch.float64 or not bool(torch.isfinite(values).all()):
        raise ValueError("CV4 values must be finite float64")
    return float(values.mean() - values.std(unbiased=True) / 2.0)


def portfolio_record(
    terms: JointTerms,
    *,
    expected_additive_gain: float,
    budget: int,
    spent: int,
    outer_sse: float,
) -> dict:
    """Validate replay against sealed additive gain and normalize one portfolio."""

    if not isinstance(terms, JointTerms):
        raise TypeError("terms must be JointTerms")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if not isinstance(spent, int) or isinstance(spent, bool) or not 0 < spent <= budget:
        raise ValueError("spent must be a positive integer no larger than budget")
    values = (expected_additive_gain, outer_sse)
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in values
    ):
        raise ValueError("expected gain and outer SSE must be finite")
    if outer_sse <= 0:
        raise ValueError("outer_sse must be positive")
    tolerance = float(PROTOCOL_CONSTANTS["numeric_relative_tolerance"]) * max(
        1.0, abs(float(expected_additive_gain))
    )
    error = abs(terms.additive_gain - float(expected_additive_gain))
    if error > tolerance:
        raise ValueError("replayed additive gain differs from the sealed state")
    scale = budget / spent / float(outer_sse)
    interaction_from_gains = terms.additive_gain - terms.joint_gain
    interaction_from_quadratics = terms.joint_quadratic - terms.diagonal_quadratic
    identity_error = abs(interaction_from_gains - interaction_from_quadratics)
    if identity_error > tolerance:
        raise ValueError("interaction identities disagree")
    return {
        "linear": terms.linear,
        "diagonal_quadratic": terms.diagonal_quadratic,
        "joint_quadratic": terms.joint_quadratic,
        "additive_gain": terms.additive_gain,
        "joint_gain": terms.joint_gain,
        "interaction_penalty": terms.interaction_penalty,
        "p_add": scale * terms.additive_gain,
        "p_joint": scale * terms.joint_gain,
        "p_int": scale * terms.interaction_penalty,
        "expected_additive_gain": float(expected_additive_gain),
        "additive_absolute_error": error,
        "interaction_identity_absolute_error": identity_error,
        "numeric_tolerance": tolerance,
        "budget": budget,
        "spent": spent,
        "outer_sse": float(outer_sse),
        "input_rows": terms.input_rows,
        "reduced_rows": terms.reduced_rows,
    }


def scene_family_reading(interaction: torch.Tensor) -> dict:
    """Apply the frozen reading to ``[selector, budget, fold]`` P_int values."""

    selectors = tuple(PROTOCOL_CONSTANTS["selectors"])
    budgets = tuple(PROTOCOL_CONSTANTS["budgets_cost_units"])
    expected = (len(selectors), len(budgets), 4)
    if not torch.is_tensor(interaction) or interaction.shape != expected:
        raise ValueError(f"interaction must have shape {list(expected)}")
    if interaction.dtype != torch.float64 or not bool(
        torch.isfinite(interaction).all()
    ):
        raise ValueError("interaction must be finite float64")

    minimum = int(PROTOCOL_CONSTANTS["minimum_directional_folds"])
    primary = selectors.index("primary")
    small = budgets.index(4096)
    large = budgets.index(16384)
    checks = {
        "large_primary_three_positive": int((interaction[primary, large] > 0).sum())
        >= minimum,
        "large_primary_cv4_positive": cv4(interaction[primary, large]) > 0,
    }
    for control_name in PROTOCOL_CONSTANTS["controls"]:
        control = selectors.index(control_name)
        large_difference = interaction[primary, large] - interaction[control, large]
        growth_difference = (
            interaction[primary, large]
            - interaction[primary, small]
            - interaction[control, large]
            + interaction[control, small]
        )
        checks[f"large_primary_beats_{control_name}_three_folds"] = int(
            (large_difference > 0).sum()
        ) >= minimum
        checks[f"large_primary_beats_{control_name}_cv4"] = cv4(large_difference) > 0
        checks[f"primary_excess_growth_beats_{control_name}_cv4"] = (
            cv4(growth_difference) > 0
        )
    return {
        "checks": checks,
        "supported": all(checks.values()),
        "large_primary_interaction_cv4": cv4(interaction[primary, large]),
    }


def overall_reading(per_scene: dict[str, dict[str, dict]]) -> dict:
    scenes = tuple(PROTOCOL_CONSTANTS["scenes"])
    families = tuple(PROTOCOL_CONSTANTS["families"])
    if set(per_scene) != set(scenes):
        raise ValueError("overall reading requires exactly the frozen scenes")
    for scene in scenes:
        if set(per_scene[scene]) != set(families):
            raise ValueError("overall reading requires exactly the frozen families")
        for family in families:
            value = per_scene[scene][family]
            if not isinstance(value, dict) or not isinstance(value.get("supported"), bool):
                raise ValueError("scene-family reading lacks a Boolean supported field")
    supported = [
        family
        for family in families
        if all(per_scene[scene][family]["supported"] for scene in scenes)
    ]
    localized = {
        scene: [family for family in families if per_scene[scene][family]["supported"]]
        for scene in scenes
    }
    return {
        "decision": "supported" if supported else "rejected",
        "supported_common_families": supported,
        "localized_supported_families": localized,
    }
