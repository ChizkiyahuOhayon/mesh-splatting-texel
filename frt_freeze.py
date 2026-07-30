"""Integrity helpers for frozen-base residual training."""

import torch


BASE_TENSOR_NAMES = (
    "vertices",
    "_triangle_indices",
    "vertex_weight",
    "_features_dc",
    "_features_rest",
    "_sigma",
)


def freeze_base_tensors(model):
    """Freeze and fingerprint every tensor that defines the base renderer."""
    fingerprint = {}
    for name in BASE_TENSOR_NAMES:
        value = getattr(model, name)
        if not torch.is_tensor(value):
            continue
        if value.is_floating_point() or value.is_complex():
            value.requires_grad_(False)
        fingerprint[name] = (value.data_ptr(), value._version, tuple(value.shape))
    return fingerprint


def assert_base_unchanged(model, fingerprint):
    """Fail if training replaced, resized, or modified any frozen base tensor."""
    for name, expected in fingerprint.items():
        value = getattr(model, name)
        observed = (value.data_ptr(), value._version, tuple(value.shape))
        if observed != expected:
            raise RuntimeError(
                f"frozen base tensor {name} changed: expected {expected}, got {observed}"
            )
