"""ADC-F0 reading: does the SH-DC gradient ratio depend on the primitive?

Everything needed to interpret a forensics report lives here and nothing else does,
so the rule can be exercised without a GPU, a checkpoint, or the CUDA extension.
The measurement that produces those reports is adc_forensics.py; the rule itself is
locked by experiments/adc_f0/protocol.md.
"""

import torch

from adc_probe import STRATA


COVARIATES = ("projected_size", "depth", "max_blending")

MIN_SURVIVAL_FRACTION = 0.8
# max/min of the per-stratum median ratios. Matches FD_FIDELITY_TOLERANCE = 0.25 in
# rits_t0_train.py so the project applies one notion of "the same ratio" throughout.
HOMOGENEITY_SPREAD = 1.25
# Catches a monotone trend too gentle to move the quintile extremes.
HOMOGENEITY_RHO = 0.3


def _average_ranks(values):
    """Ranks with ties averaged, so a covariate that saturates -- `max_blending`
    pinned at 1.0 across many faces -- is not given an arbitrary ordering."""
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(1, values.numel() + 1, dtype=values.dtype)
    unique, inverse = torch.unique(values, return_inverse=True)
    totals = torch.zeros(unique.numel(), dtype=values.dtype).scatter_add_(0, inverse, ranks)
    counts = torch.zeros(unique.numel(), dtype=values.dtype).scatter_add_(
        0, inverse, torch.ones_like(ranks)
    )
    return (totals / counts)[inverse]


def spearman(x, y):
    """Rank correlation; 0.0 when either input is constant and rho is undefined."""
    a = _average_ranks(torch.as_tensor(x, dtype=torch.float64))
    b = _average_ranks(torch.as_tensor(y, dtype=torch.float64))
    a = a - a.mean()
    b = b - b.mean()
    scale = a.norm() * b.norm()
    return float(a @ b / scale) if scale > 0 else 0.0


def stratum_medians(values, ratios, strata=STRATA):
    """Median ratio within each equal-count quantile bin of `values`, in bin order."""
    values = torch.as_tensor(values, dtype=torch.float64)
    ratios = torch.as_tensor(ratios, dtype=torch.float64)
    order = torch.argsort(values)
    return [
        float(chunk.median())
        for chunk in torch.chunk(ratios[order], strata)
        if chunk.numel() > 0
    ]


def spread(medians):
    """max/min of the stratum medians, or None when that ratio is meaningless.

    A non-positive median means the analytic gradient and the central difference
    disagreed in sign somewhere in that bin. Returning None rather than infinity
    keeps the record serialisable under `allow_nan=False`, which is how every
    results file in this project is written.
    """
    if not medians or min(medians) <= 0:
        return None
    return max(medians) / min(medians)


def _offenders(report):
    """Covariates on which this view's ratios are not flat."""
    found = {}
    for covariate in COVARIATES:
        ratio_spread = spread(report["stratum_medians"][covariate])
        rho = report["spearman"][covariate]
        if ratio_spread is None or ratio_spread > HOMOGENEITY_SPREAD or abs(rho) > HOMOGENEITY_RHO:
            found[covariate] = {"spread": ratio_spread, "spearman": rho}
    return found


def read(view_reports):
    """Turn per-view probe tables into the protocol's reading.

    Homogeneity must hold on every covariate and every view. Views that disagree are
    reported as disagreeing rather than pooled: two views are replication, and
    averaging them would hide exactly the failure replication exists to catch.
    """
    if not view_reports:
        return {"reading": "INCONCLUSIVE", "reason": "no views measured"}
    if any(report["survival_fraction"] < MIN_SURVIVAL_FRACTION for report in view_reports):
        return {"reading": "INCONCLUSIVE", "reason": "too few probes survived the rung check"}
    if not all(report["deterministic_forward"] for report in view_reports):
        return {"reading": "INCONCLUSIVE", "reason": "forward pass was not deterministic"}

    evidence = [
        {"view": report["view"], "offenders": _offenders(report)} for report in view_reports
    ]
    verdicts = {"HETEROGENEOUS" if item["offenders"] else "HOMOGENEOUS" for item in evidence}

    if len(verdicts) > 1:
        return {"reading": "VIEW_DEPENDENT", "per_view": evidence}
    if verdicts == {"HOMOGENEOUS"}:
        return {"reading": "HOMOGENEOUS", "per_view": evidence}

    covariate, detail, view = max(
        (
            (covariate, detail, item["view"])
            for item in evidence
            for covariate, detail in item["offenders"].items()
        ),
        key=lambda row: abs(row[1]["spearman"]),
    )
    return {
        "reading": "HETEROGENEOUS",
        "per_view": evidence,
        "strongest_trend": {"view": view, "covariate": covariate, **detail},
    }
