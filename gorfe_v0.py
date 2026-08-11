"""Duplicate-safe sufficient statistics for the GoRFE-V0 integrity gate."""

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class CameraDesignRows:
    name: str
    fold: int
    residual_pixel_ids: torch.Tensor
    residuals: torch.Tensor
    pixel_ids: torch.Tensor
    group_ids: torch.Tensor
    features: torch.Tensor


@dataclass(frozen=True)
class FoldStatistics:
    gram: torch.Tensor
    rhs: torch.Tensor
    support_rss: torch.Tensor
    support_pixels: torch.Tensor


@dataclass(frozen=True)
class StreamDiagnostics:
    cameras: int
    input_rows: int
    reduced_rows: int
    max_camera_input_rows: int
    max_camera_reduced_rows: int
    estimated_peak_temporary_bytes: int


@dataclass(frozen=True)
class HeldoutFit:
    ridge: torch.Tensor
    coefficients: torch.Tensor
    signed_gain: torch.Tensor


def _require_integer_vector(value, name):
    if not torch.is_tensor(value) or value.ndim != 1:
        raise ValueError(f"{name} must be a rank-one tensor")
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64 indices")


def _validate_camera_rows(rows, group_count, fold_count, feature_dim, device):
    if not isinstance(rows.name, str) or not rows.name:
        raise ValueError("camera name must be a nonempty string")
    if not 0 <= rows.fold < fold_count:
        raise ValueError(f"camera fold must be in [0, {fold_count})")
    _require_integer_vector(rows.residual_pixel_ids, "residual_pixel_ids")
    _require_integer_vector(rows.pixel_ids, "pixel_ids")
    _require_integer_vector(rows.group_ids, "group_ids")
    if rows.residuals.shape != (rows.residual_pixel_ids.numel(), 3):
        raise ValueError("residuals must have shape [number of residual pixels, 3]")
    if rows.features.shape != (rows.pixel_ids.numel(), feature_dim):
        raise ValueError(f"features must have shape [number of rows, {feature_dim}]")
    if rows.group_ids.numel() != rows.pixel_ids.numel():
        raise ValueError("pixel_ids and group_ids must have the same number of rows")
    if rows.residuals.dtype != torch.float64 or rows.features.dtype != torch.float64:
        raise TypeError("residuals and features must use float64")
    tensors = (
        rows.residual_pixel_ids,
        rows.residuals,
        rows.pixel_ids,
        rows.group_ids,
        rows.features,
    )
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all camera tensors must be on the accumulator device")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in (rows.residuals, rows.features)):
        raise ValueError("residuals and features must be finite")
    if bool((rows.residual_pixel_ids < 0).any()) or bool((rows.pixel_ids < 0).any()):
        raise ValueError("pixel ids must be nonnegative")
    if bool((rows.group_ids < 0).any()) or bool((rows.group_ids >= group_count).any()):
        raise ValueError(f"group ids must be in [0, {group_count})")
    if torch.unique(rows.residual_pixel_ids).numel() != rows.residual_pixel_ids.numel():
        raise ValueError("residual_pixel_ids must be unique within a camera")


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _reduce_camera_rows(rows, group_count):
    """Reduce duplicate `(pixel, group)` rows without a dense group matrix."""
    row_count = rows.pixel_ids.numel()
    if row_count == 0:
        empty_ids = torch.empty(0, dtype=torch.int64, device=rows.features.device)
        empty_features = torch.empty(
            (0, rows.features.shape[1]), dtype=torch.float64, device=rows.features.device
        )
        return empty_ids, empty_ids, empty_features

    pixel_ids = rows.pixel_ids.to(torch.int64)
    group_ids = rows.group_ids.to(torch.int64)
    max_pixel = int(pixel_ids.max())
    if max_pixel > (torch.iinfo(torch.int64).max - (group_count - 1)) // group_count:
        raise OverflowError("pixel/group key exceeds int64")
    keys = pixel_ids * group_count + group_ids
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    reduced = torch.zeros(
        (unique_keys.numel(), rows.features.shape[1]),
        dtype=torch.float64,
        device=rows.features.device,
    )
    reduced.index_add_(0, inverse, rows.features)
    supported = torch.any(reduced != 0, dim=1)
    unique_keys = unique_keys[supported]
    reduced = reduced[supported]
    return unique_keys // group_count, unique_keys % group_count, reduced


class CameraStreamingFoldAccumulator:
    """Accumulate sparse GoRFE statistics one complete camera at a time."""

    def __init__(self, group_count, fold_count, feature_dim, *, device="cpu"):
        if group_count < 1:
            raise ValueError("group_count must be positive")
        if fold_count < 2:
            raise ValueError("fold_count must be at least two")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.group_count = int(group_count)
        self.fold_count = int(fold_count)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self._gram = torch.zeros(
            (group_count, fold_count, feature_dim, feature_dim),
            dtype=torch.float64,
            device=self.device,
        )
        self._rhs = torch.zeros(
            (group_count, fold_count, feature_dim, 3),
            dtype=torch.float64,
            device=self.device,
        )
        self._rss = torch.zeros(
            (group_count, fold_count), dtype=torch.float64, device=self.device
        )
        self._counts = torch.zeros(
            (group_count, fold_count), dtype=torch.int64, device=self.device
        )
        self._camera_names = set()
        self._input_rows = 0
        self._reduced_rows = 0
        self._max_input_rows = 0
        self._max_reduced_rows = 0
        self._estimated_peak_bytes = 0

    def add_camera(self, rows, *, chunk_size=None):
        """Add one camera; `chunk_size` controls sparse ingestion only."""
        _validate_camera_rows(
            rows, self.group_count, self.fold_count, self.feature_dim, self.device
        )
        if rows.name in self._camera_names:
            raise ValueError(f"camera {rows.name!r} was added more than once")
        row_count = rows.pixel_ids.numel()
        if chunk_size is None:
            chunk_size = max(row_count, 1)
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")

        pixel_chunks = []
        group_chunks = []
        feature_chunks = []
        for start in range(0, row_count, chunk_size):
            stop = min(start + chunk_size, row_count)
            pixel_chunks.append(rows.pixel_ids[start:stop])
            group_chunks.append(rows.group_ids[start:stop])
            feature_chunks.append(rows.features[start:stop])
        if row_count:
            buffered = CameraDesignRows(
                rows.name,
                rows.fold,
                rows.residual_pixel_ids,
                rows.residuals,
                torch.cat(pixel_chunks),
                torch.cat(group_chunks),
                torch.cat(feature_chunks),
            )
        else:
            buffered = rows

        reduced_pixels, reduced_groups, reduced_features = _reduce_camera_rows(
            buffered, self.group_count
        )
        reduced_count = reduced_pixels.numel()
        raw_bytes = sum(
            _tensor_bytes(tensor)
            for tensor in (buffered.pixel_ids, buffered.group_ids, buffered.features)
        )
        residual_bytes = _tensor_bytes(rows.residual_pixel_ids) + _tensor_bytes(
            rows.residuals
        )
        # Tensor-only upper estimate: caller rows plus concatenated sparse rows,
        # sorted residual copies, and (below) all duplicate-reduction products.
        self._estimated_peak_bytes = max(
            self._estimated_peak_bytes, 2 * raw_bytes + 2 * residual_bytes
        )
        if reduced_count:
            if rows.residual_pixel_ids.numel() == 0:
                raise ValueError(
                    f"contribution references missing residual pixel {int(reduced_pixels[0])}"
                )
            order = torch.argsort(rows.residual_pixel_ids)
            sorted_pixels = rows.residual_pixel_ids[order].to(torch.int64)
            sorted_residuals = rows.residuals[order]
            locations = torch.searchsorted(sorted_pixels, reduced_pixels)
            in_range = locations < sorted_pixels.numel()
            safe_locations = locations.clamp_max(sorted_pixels.numel() - 1)
            found = in_range & (sorted_pixels[safe_locations] == reduced_pixels)
            if not bool(found.all()):
                missing = int(reduced_pixels[torch.nonzero(~found, as_tuple=False)[0, 0]])
                raise ValueError(f"contribution references missing residual pixel {missing}")
            residual = sorted_residuals[locations]

            gram_rows = torch.einsum("mi,mj->mij", reduced_features, reduced_features)
            rhs_rows = torch.einsum("mi,mc->mic", reduced_features, residual)
            rss_rows = residual.square().sum(dim=1)
            fold_gram = self._gram[:, rows.fold]
            fold_rhs = self._rhs[:, rows.fold]
            fold_rss = self._rss[:, rows.fold]
            fold_counts = self._counts[:, rows.fold]
            fold_gram.index_add_(0, reduced_groups, gram_rows)
            fold_rhs.index_add_(0, reduced_groups, rhs_rows)
            fold_rss.index_add_(0, reduced_groups, rss_rows)
            fold_counts.index_add_(
                0, reduced_groups, torch.ones_like(reduced_groups, dtype=torch.int64)
            )

            reduced_bytes = sum(
                _tensor_bytes(tensor)
                for tensor in (
                    reduced_pixels,
                    reduced_groups,
                    reduced_features,
                    gram_rows,
                    rhs_rows,
                    rss_rows,
                    locations,
                )
            )
            self._estimated_peak_bytes = max(
                self._estimated_peak_bytes,
                2 * raw_bytes + reduced_bytes + 2 * residual_bytes,
            )

        self._camera_names.add(rows.name)
        self._input_rows += row_count
        self._reduced_rows += reduced_count
        self._max_input_rows = max(self._max_input_rows, row_count)
        self._max_reduced_rows = max(self._max_reduced_rows, reduced_count)

    def statistics(self):
        return FoldStatistics(
            self._gram.clone(), self._rhs.clone(), self._rss.clone(), self._counts.clone()
        )

    def diagnostics(self):
        return StreamDiagnostics(
            cameras=len(self._camera_names),
            input_rows=self._input_rows,
            reduced_rows=self._reduced_rows,
            max_camera_input_rows=self._max_input_rows,
            max_camera_reduced_rows=self._max_reduced_rows,
            estimated_peak_temporary_bytes=self._estimated_peak_bytes,
        )


def streaming_fold_statistics(
    cameras: Iterable[CameraDesignRows],
    group_count,
    fold_count,
    feature_dim,
    *,
    chunk_size=None,
    device="cpu",
):
    accumulator = CameraStreamingFoldAccumulator(
        group_count, fold_count, feature_dim, device=device
    )
    for camera in cameras:
        accumulator.add_camera(camera, chunk_size=chunk_size)
    return accumulator.statistics(), accumulator.diagnostics()


def dense_fold_statistics(cameras, group_count, fold_count, feature_dim):
    """Independent V0 oracle using an explicit pixel-by-group dense design."""
    gram = torch.zeros((group_count, fold_count, feature_dim, feature_dim), dtype=torch.float64)
    rhs = torch.zeros((group_count, fold_count, feature_dim, 3), dtype=torch.float64)
    rss = torch.zeros((group_count, fold_count), dtype=torch.float64)
    counts = torch.zeros((group_count, fold_count), dtype=torch.int64)
    names = set()
    for rows in cameras:
        _validate_camera_rows(rows, group_count, fold_count, feature_dim, torch.device("cpu"))
        if rows.name in names:
            raise ValueError(f"camera {rows.name!r} was added more than once")
        names.add(rows.name)
        residual_lookup = {
            int(pixel): index for index, pixel in enumerate(rows.residual_pixel_ids.tolist())
        }
        dense = torch.zeros(
            (rows.residual_pixel_ids.numel(), group_count, feature_dim), dtype=torch.float64
        )
        for index in range(rows.pixel_ids.numel()):
            pixel = int(rows.pixel_ids[index])
            if pixel not in residual_lookup:
                raise ValueError(f"contribution references missing residual pixel {pixel}")
            dense[residual_lookup[pixel], int(rows.group_ids[index])] += rows.features[index]

        for group in range(group_count):
            supported = torch.any(dense[:, group] != 0, dim=1)
            features = dense[supported, group]
            residual = rows.residuals[supported]
            count = features.shape[0]
            if not count:
                continue
            block_design = torch.zeros(
                (count * 3, feature_dim * 3), dtype=torch.float64
            )
            for pixel_row in range(count):
                for channel in range(3):
                    block_design[
                        pixel_row * 3 + channel, channel::3
                    ] = features[pixel_row]
            full_gram = block_design.T @ block_design
            full_rhs = block_design.T @ residual.reshape(-1)
            reference_gram = full_gram[0::3, 0::3]
            for channel in range(3):
                if not torch.allclose(
                    full_gram[channel::3, channel::3],
                    reference_gram,
                    atol=1e-13,
                    rtol=1e-13,
                ):
                    raise AssertionError("RGB block design has inconsistent diagonal blocks")
            off_channel = torch.ones_like(full_gram, dtype=torch.bool)
            for channel in range(3):
                off_channel[channel::3, channel::3] = False
            if bool((full_gram[off_channel] != 0).any()):
                raise AssertionError("RGB block design has a nonzero off-channel block")
            gram[group, rows.fold] += reference_gram
            rhs[group, rows.fold] += full_rhs.reshape(feature_dim, 3)
            rss[group, rows.fold] += residual.square().sum()
            counts[group, rows.fold] += count
    return FoldStatistics(gram, rhs, rss, counts)


def heldout_signed_gain(statistics, *, ridge_fraction=1e-3):
    """Fit on complementary folds and evaluate exact signed held-out gain."""
    gram = statistics.gram
    rhs = statistics.rhs
    if gram.ndim != 4 or gram.shape[-1] != gram.shape[-2]:
        raise ValueError("gram must have shape [groups, folds, Q, Q]")
    groups, folds, feature_dim, _ = gram.shape
    if folds < 2 or rhs.shape != (groups, folds, feature_dim, 3):
        raise ValueError("rhs must have shape [groups, folds, Q, 3]")
    if statistics.support_rss.shape != (groups, folds):
        raise ValueError("support_rss must have shape [groups, folds]")
    if statistics.support_pixels.shape != (groups, folds):
        raise ValueError("support_pixels must have shape [groups, folds]")
    if gram.dtype != torch.float64 or rhs.dtype != torch.float64:
        raise TypeError("gram and rhs must use float64")
    if not ridge_fraction > 0:
        raise ValueError("ridge_fraction must be positive")

    train_gram = gram.sum(dim=1, keepdim=True) - gram
    train_rhs = rhs.sum(dim=1, keepdim=True) - rhs
    ridge = train_gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    ridge = ridge * (ridge_fraction / feature_dim)
    if not bool(torch.isfinite(ridge).all()) or bool((ridge <= 0).any()):
        raise ValueError("every held-out fit needs a positive finite ridge scale")
    eye = torch.eye(feature_dim, dtype=torch.float64, device=gram.device)
    coefficients = torch.linalg.solve(
        train_gram + ridge[..., None, None] * eye, train_rhs
    )
    linear = (coefficients * rhs).sum(dim=(-2, -1))
    quadratic = torch.einsum("gkqc,gkqr,gkrc->gk", coefficients, gram, coefficients)
    gain = 2.0 * linear - quadratic
    return HeldoutFit(ridge, coefficients, gain)
