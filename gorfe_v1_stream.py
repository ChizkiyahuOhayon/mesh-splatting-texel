"""Camera-streaming sufficient statistics for the GoRFE-V1 real-scene gate."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CarrierStatistics:
    gram: torch.Tensor
    rhs: torch.Tensor
    support_rss: torch.Tensor
    support_pixels: torch.Tensor
    support_cameras: torch.Tensor


@dataclass(frozen=True)
class StreamStatistics:
    dc: CarrierStatistics
    sh1: CarrierStatistics
    fold_full_rss: torch.Tensor


@dataclass(frozen=True)
class CameraReduction:
    input_rows: int
    reduced_rows: int
    dc_support_rows: int
    sh1_support_rows: int


def _require_vector(tensor, name):
    if not torch.is_tensor(tensor) or tensor.ndim != 1:
        raise ValueError(f"{name} must be a rank-one tensor")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64")


def reduce_camera_design(pixel_ids, group_ids, features, group_count):
    """Sum every final-resolution ``(pixel, group)`` before any outer product."""
    _require_vector(pixel_ids, "pixel_ids")
    _require_vector(group_ids, "group_ids")
    if features.ndim != 2 or features.shape != (pixel_ids.numel(), 4):
        raise ValueError("features must have shape [number of rows, 4]")
    if group_ids.numel() != pixel_ids.numel():
        raise ValueError("pixel_ids and group_ids must have equal length")
    if features.dtype not in (torch.float32, torch.float64):
        raise TypeError("features must use float32 or float64")
    if len({pixel_ids.device, group_ids.device, features.device}) != 1:
        raise ValueError("camera design tensors must share one device")
    if group_count < 1:
        raise ValueError("group_count must be positive")
    if bool((pixel_ids < 0).any()):
        raise ValueError("pixel ids must be nonnegative")
    if bool((group_ids < 0).any()) or bool((group_ids >= group_count).any()):
        raise ValueError(f"group ids must lie in [0, {group_count})")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features must be finite")
    if not pixel_ids.numel():
        empty = torch.empty(0, dtype=torch.int64, device=features.device)
        return empty, empty, torch.empty((0, 4), dtype=torch.float64, device=features.device)

    pixels = pixel_ids.to(torch.int64)
    groups = group_ids.to(torch.int64)
    maximum_pixel = int(pixels.max())
    maximum_key = torch.iinfo(torch.int64).max
    if maximum_pixel > (maximum_key - (group_count - 1)) // group_count:
        raise OverflowError("pixel/group key exceeds int64")
    keys = pixels * group_count + groups
    order = torch.argsort(keys, stable=True)
    keys = keys[order]
    values = features[order].to(torch.float64)
    unique_keys, inverse = torch.unique_consecutive(keys, return_inverse=True)
    reduced = torch.zeros(
        (unique_keys.numel(), 4), dtype=torch.float64, device=features.device
    )
    reduced.index_add_(0, inverse, values)
    return unique_keys // group_count, unique_keys % group_count, reduced


class GoRFEV1Accumulator:
    """Accumulate both DC and SH1 statistics, one complete camera at a time."""

    def __init__(self, group_count, fold_count=4, *, device="cpu", raw_row_limit=2**31 - 1):
        if group_count < 1:
            raise ValueError("group_count must be positive")
        if fold_count < 2:
            raise ValueError("fold_count must be at least two")
        self.group_count = int(group_count)
        self.fold_count = int(fold_count)
        self.device = torch.device(device)
        self.raw_row_limit = int(raw_row_limit)
        self._families = {
            "dc": self._allocate(1),
            "sh1": self._allocate(3),
        }
        self._fold_full_rss = torch.zeros(
            fold_count, dtype=torch.float64, device=self.device
        )
        self._camera_names = set()
        self._diagnostics = []

    def _allocate(self, feature_dim):
        return {
            "gram": torch.zeros(
                (self.group_count, self.fold_count, feature_dim, feature_dim),
                dtype=torch.float64,
                device=self.device,
            ),
            "rhs": torch.zeros(
                (self.group_count, self.fold_count, feature_dim, 3),
                dtype=torch.float64,
                device=self.device,
            ),
            "rss": torch.zeros(
                (self.group_count, self.fold_count),
                dtype=torch.float64,
                device=self.device,
            ),
            "pixels": torch.zeros(
                (self.group_count, self.fold_count),
                dtype=torch.int64,
                device=self.device,
            ),
            "cameras": torch.zeros(
                (self.group_count, self.fold_count),
                dtype=torch.int64,
                device=self.device,
            ),
        }

    def add_camera(
        self,
        *,
        name,
        fold,
        pixel_count,
        pixel_ids,
        group_ids,
        features,
        residuals=None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("camera name must be nonempty")
        if name in self._camera_names:
            raise ValueError(f"camera {name!r} was added twice")
        if not 0 <= fold < self.fold_count:
            raise ValueError(f"fold must lie in [0, {self.fold_count})")
        if not isinstance(pixel_count, int) or pixel_count < 1:
            raise ValueError("pixel_count must be positive")
        if pixel_ids.numel() > self.raw_row_limit:
            raise OverflowError("camera raw-row limit exceeded")
        if features.device != self.device:
            raise ValueError("features and accumulator must share one device")
        if pixel_ids.numel() and int(pixel_ids.max()) >= pixel_count:
            raise ValueError("a design row references a pixel outside the output image")
        if residuals is not None:
            if residuals.shape != (pixel_count, 3) or residuals.dtype != torch.float64:
                raise ValueError("residuals must be float64 [pixel_count, 3]")
            if residuals.device != self.device or not bool(torch.isfinite(residuals).all()):
                raise ValueError("residuals must be finite and on the accumulator device")
            self._fold_full_rss[fold] += residuals.square().sum()

        reduced_pixels, reduced_groups, reduced = reduce_camera_design(
            pixel_ids, group_ids, features, self.group_count
        )
        support_rows = {}
        for family, columns in (("dc", slice(0, 1)), ("sh1", slice(1, 4))):
            design = reduced[:, columns]
            supported = torch.any(design != 0, dim=1)
            family_groups = reduced_groups[supported]
            family_pixels = reduced_pixels[supported]
            design = design[supported]
            support_rows[family] = int(design.shape[0])
            if not design.numel():
                continue
            target = self._families[family]
            gram_rows = torch.einsum("mi,mj->mij", design, design)
            target["gram"][:, fold].index_add_(0, family_groups, gram_rows)
            target["pixels"][:, fold].index_add_(
                0, family_groups, torch.ones_like(family_groups, dtype=torch.int64)
            )
            camera_groups = torch.unique(family_groups, sorted=True)
            target["cameras"][:, fold].index_add_(
                0, camera_groups, torch.ones_like(camera_groups, dtype=torch.int64)
            )
            if residuals is not None:
                residual = residuals[family_pixels]
                rhs_rows = torch.einsum("mi,mc->mic", design, residual)
                target["rhs"][:, fold].index_add_(0, family_groups, rhs_rows)
                target["rss"][:, fold].index_add_(
                    0, family_groups, residual.square().sum(dim=1)
                )

        diagnostic = CameraReduction(
            int(pixel_ids.numel()),
            int(reduced.shape[0]),
            support_rows["dc"],
            support_rows["sh1"],
        )
        self._diagnostics.append((name, int(fold), diagnostic))
        self._camera_names.add(name)
        return diagnostic

    def statistics(self):
        def freeze(family):
            value = self._families[family]
            return CarrierStatistics(
                value["gram"].clone(),
                value["rhs"].clone(),
                value["rss"].clone(),
                value["pixels"].clone(),
                value["cameras"].clone(),
            )

        return StreamStatistics(freeze("dc"), freeze("sh1"), self._fold_full_rss.clone())

    def diagnostics(self):
        return tuple(self._diagnostics)
