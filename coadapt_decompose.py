"""Pure tensor helpers for the COADAPT-G0 checkpoint decomposition."""

import torch


def texel_variants(texels):
    """Return carriers that remove all texels or only within-face variation."""
    if texels.ndim != 3 or texels.shape[-1] != 3:
        raise ValueError(f"expected texels shaped [faces, cells, 3], got {tuple(texels.shape)}")
    face_mean = texels.mean(dim=1, keepdim=True)
    return {
        "zero": torch.zeros_like(texels),
        "face_mean": face_mean.expand_as(texels),
    }


def recovery_fraction(reference, fixed, candidate, higher_is_better):
    """Fraction of a fixed-model regression recovered by a candidate."""
    regression = reference - fixed if higher_is_better else fixed - reference
    if regression <= 0:
        raise ValueError("the fixed variant must regress relative to the reference")
    recovered = candidate - fixed if higher_is_better else fixed - candidate
    return recovered / regression
