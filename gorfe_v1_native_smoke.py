"""GPU equivalence gate for the GoRFE-V1 forward-state exporter."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from diff_triangle_rasterization import (
    TriangleRasterizationSettings,
    TriangleRasterizer,
)
from gorfe_v1_stream import reduce_camera_design
from gorfe_v1_io import write_json_new
from utils.graphics_utils import getProjectionMatrix


OUTPUT_SCALING = 4
HIGH_SIZE = 64
OUTPUT_SIZE = HIGH_SIZE // OUTPUT_SCALING
RECONSTRUCTION_ATOL = 5e-5
RECONSTRUCTION_RTOL = 5e-4


def _sha256(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def _rasterizer(device):
    field_of_view = 1.0
    settings = TriangleRasterizationSettings(
        image_height=HIGH_SIZE,
        image_width=HIGH_SIZE,
        tanfovx=math.tan(field_of_view / 2),
        tanfovy=math.tan(field_of_view / 2),
        bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, dtype=torch.float32, device=device),
        projmatrix=getProjectionMatrix(
            0.01, 100.0, field_of_view, field_of_view
        ).transpose(0, 1).to(device),
        sh_degree=0,
        campos=torch.tensor([0.17, -0.11, 0.03], dtype=torch.float32, device=device),
        prefiltered=False,
        debug=False,
    )
    return TriangleRasterizer(settings)


def _fixture(device):
    # Two depth-separated triangles intentionally project to the same pixels and
    # share group zero.  They exercise the cross terms that disappear if a
    # fragment outer product is formed before duplicate reduction.
    vertices = torch.tensor(
        [
            [-0.62, -0.52, 2.0],
            [0.62, -0.52, 2.0],
            [0.0, 0.64, 2.0],
            [-0.775, -0.65, 2.5],
            [0.775, -0.65, 2.5],
            [0.0, 0.80, 2.5],
        ],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    vertex_weights = torch.full((6,), 0.73, dtype=torch.float32, device=device)
    colors = torch.tensor(
        [
            [0.20, 0.34, 0.42],
            [0.43, 0.18, 0.12],
            [0.11, 0.51, 0.25],
            [0.28, 0.16, 0.38],
            [0.17, 0.41, 0.31],
            [0.37, 0.27, 0.14],
        ],
        dtype=torch.float32,
        device=device,
    )
    face_edge_ids = torch.tensor(
        [[0, -1, -1], [0, -1, -1]], dtype=torch.int32, device=device
    )
    scaling = torch.zeros(2, dtype=torch.float32, device=device)
    return vertices, faces, vertex_weights, colors, face_edge_ids, scaling


def _render(
    rasterizer, *, edge_details=None, export=False, blank=False, candidate_map=None
):
    device = rasterizer.raster_settings.bg.device
    vertices, faces, weights, colors, face_edge_ids, scaling = _fixture(device)
    arguments = dict(
        vertices=vertices,
        triangles_indices=faces,
        vertex_weights=weights,
        sigma=0.0001,
        scaling=scaling,
        colors_precomp=colors,
    )
    if edge_details is not None:
        arguments.update(edge_details=edge_details, face_edge_ids=face_edge_ids)
    if not export:
        return rasterizer(**arguments)
    if blank and candidate_map is not None:
        raise ValueError("blank and candidate_map are mutually exclusive")
    if candidate_map is None:
        candidate_map = torch.full_like(face_edge_ids, -1) if blank else face_edge_ids
    return rasterizer.forward_with_gorfe_design(
        **arguments,
        gorfe_face_edge_ids=candidate_map,
        gorfe_edge_count=0 if blank else 1,
        output_height=OUTPUT_SIZE,
        output_width=OUTPUT_SIZE,
        output_scaling=OUTPUT_SCALING,
    )


def _downsample(image):
    return F.interpolate(
        image.unsqueeze(0),
        size=(OUTPUT_SIZE, OUTPUT_SIZE),
        mode="area",
    ).squeeze(0)


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("GoRFE-V1 native smoke requires CUDA")
    device = torch.device("cuda:0")
    rasterizer = _rasterizer(device)

    ordinary = _render(rasterizer)
    exported = _render(rasterizer, export=True)
    blank = _render(rasterizer, export=True, blank=True)
    front_candidate_map = torch.tensor(
        [[0, -1, -1], [-1, -1, -1]], dtype=torch.int32, device=device
    )
    back_candidate_map = torch.tensor(
        [[-1, -1, -1], [0, -1, -1]], dtype=torch.int32, device=device
    )
    front_exported = _render(
        rasterizer, export=True, candidate_map=front_candidate_map
    )
    back_exported = _render(
        rasterizer, export=True, candidate_map=back_candidate_map
    )
    parent_high = ordinary[0]
    exported_parent_high = exported[0]
    pixel_ids, group_ids, raw_features, diagnostics = exported[6:]

    parent_output_names = (
        "color",
        "radii",
        "scaling",
        "depth",
        "max_blending",
        "was_rendered",
    )
    parent_output_bitwise = {
        name: bool(torch.equal(ordinary[index], exported[index]))
        for index, name in enumerate(parent_output_names)
    }
    parent_output_hashes = {
        name: {
            "ordinary": _sha256(ordinary[index]),
            "exported": _sha256(exported[index]),
        }
        for index, name in enumerate(parent_output_names)
    }
    parent_alpha = ordinary[3][1]
    parent_face_id = ordinary[3][6]
    no_contribution = parent_alpha == 0
    valid_contribution = parent_alpha > 0
    face_id_sentinel_is_exact = bool(
        no_contribution.any()
        and valid_contribution.any()
        and torch.all(parent_face_id[no_contribution] == -1)
        and torch.all(parent_face_id[valid_contribution] >= 0)
        and torch.all(parent_face_id[valid_contribution] < 2)
    )

    reduced_pixels, reduced_groups, reduced_features = reduce_camera_design(
        pixel_ids, group_ids, raw_features, 1
    )
    coefficient = torch.tensor(
        [
            [0.061, -0.037, 0.024],
            [0.093, -0.048, 0.031],
            [-0.055, 0.071, 0.042],
            [0.047, 0.018, -0.066],
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    active_high = _render(rasterizer, edge_details=coefficient)[0]
    native_change = (_downsample(active_high) - _downsample(parent_high)).permute(1, 2, 0)
    reconstructed = torch.zeros(
        (OUTPUT_SIZE * OUTPUT_SIZE, 3), dtype=torch.float64, device=device
    )
    reconstructed.index_add_(
        0,
        reduced_pixels,
        torch.einsum(
            "mq,mqc->mc",
            reduced_features,
            coefficient[reduced_groups].to(torch.float64),
        ),
    )
    native_flat = native_change.reshape(-1, 3).to(torch.float64)
    reconstruction_error = (reconstructed - native_flat).abs()
    reconstruction_scale = torch.maximum(
        reconstructed.abs(), native_flat.abs()
    ).clamp_min(1.0)

    keys = pixel_ids.to(torch.int64) + group_ids.to(torch.int64) * (
        OUTPUT_SIZE * OUTPUT_SIZE
    )
    duplicate_rows = int(keys.numel() - torch.unique(keys).numel())
    correct_gram = reduced_features.T @ reduced_features
    naive_gram = raw_features.to(torch.float64).T @ raw_features.to(torch.float64)

    front_pixels, front_groups, front_features = reduce_camera_design(
        *front_exported[6:9], 1
    )
    back_pixels, back_groups, back_features = reduce_camera_design(
        *back_exported[6:9], 1
    )
    dense_shape = (OUTPUT_SIZE * OUTPUT_SIZE, 4)

    def dense_design(pixels, groups, features):
        if groups.numel() and not bool((groups == 0).all()):
            raise RuntimeError("single-group depth probe produced a nonzero group id")
        dense = torch.zeros(dense_shape, dtype=torch.float64, device=device)
        dense.index_add_(0, pixels, features)
        return dense

    full_dense = dense_design(reduced_pixels, reduced_groups, reduced_features)
    front_dense = dense_design(front_pixels, front_groups, front_features)
    back_dense = dense_design(back_pixels, back_groups, back_features)
    shared_depth_pixels = front_pixels[torch.isin(front_pixels, back_pixels)]
    if shared_depth_pixels.numel():
        front_shared_nonzero = torch.any(
            front_dense[shared_depth_pixels] != 0, dim=1
        )
        back_shared_nonzero = torch.any(back_dense[shared_depth_pixels] != 0, dim=1)
        both_depths_nonzero = bool(
            torch.all(front_shared_nonzero & back_shared_nonzero)
        )
        front_shared_max = float(front_dense[shared_depth_pixels].abs().max())
        back_shared_max = float(back_dense[shared_depth_pixels].abs().max())
    else:
        both_depths_nonzero = False
        front_shared_max = 0.0
        back_shared_max = 0.0
    depth_layer_sum_error = float(
        (full_dense - front_dense - back_dense).abs().max()
    )

    generator = torch.Generator(device="cpu").manual_seed(20260811)
    target_residual = torch.randn(
        (OUTPUT_SIZE * OUTPUT_SIZE, 3), generator=generator, dtype=torch.float64
    ).to(device)
    variable = torch.zeros((4, 3), dtype=torch.float64, device=device, requires_grad=True)
    correction = reduced_features @ variable
    loss = (correction - target_residual[reduced_pixels]).square().sum()
    (gradient,) = torch.autograd.grad(loss, variable)
    rhs = reduced_features.T @ target_residual[reduced_pixels]
    gradient_error = float((gradient + 2.0 * rhs).abs().max())

    mismatch_keys = (
        "replay_transmittance_mismatch_pixels",
        "replay_last_contributor_mismatch_pixels",
        "count_write_mismatch_pixels",
        "write_overflow_rows",
    )
    checks = {
        "exporter_off_all_six_parent_outputs_are_bitwise_unchanged": all(
            parent_output_bitwise.values()
        ),
        "background_face_ids_use_the_negative_one_sentinel": (
            face_id_sentinel_is_exact
        ),
        "blank_candidate_map_exports_zero_rows": all(
            tensor.numel() == 0 for tensor in blank[6:9]
        ),
        "replay_diagnostics_are_exact": all(
            diagnostics[name] == 0 for name in mismatch_keys
        ),
        "count_write_and_forward_fragment_totals_match": (
            diagnostics["count_alpha_accepted_fragments"]
            == diagnostics["write_alpha_accepted_fragments"]
            == diagnostics["forward_alpha_accepted_fragments"]
            and diagnostics["count_blended_fragments"]
            == diagnostics["write_blended_fragments"]
        ),
        "multi_fragment_and_subpixel_duplicates_are_present": duplicate_rows > 0,
        "both_depth_layers_contribute_to_the_same_reduced_keys": (
            shared_depth_pixels.numel() > 0 and both_depths_nonzero
        ),
        "separate_depth_layer_designs_sum_to_the_full_design": bool(
            torch.allclose(
                full_dense,
                front_dense + back_dense,
                atol=1e-12,
                rtol=1e-12,
            )
        ),
        "duplicate_reduction_changes_the_gram": float(
            (correct_gram - naive_gram).abs().max()
        ) > 1e-8,
        "sparse_design_reconstructs_native_area_downsampled_carrier": bool(
            torch.allclose(
                reconstructed,
                native_flat,
                atol=RECONSTRUCTION_ATOL,
                rtol=RECONSTRUCTION_RTOL,
            )
        ),
        "squared_loss_gradient_equals_negative_two_rhs": gradient_error <= 1e-12,
        "ids_and_features_have_locked_contract": (
            pixel_ids.dtype == torch.int32
            and group_ids.dtype == torch.int32
            and raw_features.dtype == torch.float32
            and raw_features.shape == (pixel_ids.numel(), 4)
            and bool((pixel_ids >= 0).all())
            and bool((pixel_ids < OUTPUT_SIZE * OUTPUT_SIZE).all())
            and bool((group_ids == 0).all())
            and bool(torch.isfinite(raw_features).all())
        ),
    }
    result = {
        "experiment": "GoRFE-V1-native",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "checks": checks,
        "diagnostics": diagnostics,
        "parent_output_bitwise_equal": parent_output_bitwise,
        "parent_output_sha256": parent_output_hashes,
        "raw_rows": int(pixel_ids.numel()),
        "reduced_rows": int(reduced_pixels.numel()),
        "duplicate_rows": duplicate_rows,
        "max_reconstruction_absolute_error": float(reconstruction_error.max()),
        "max_reconstruction_scaled_error": float(
            (reconstruction_error / reconstruction_scale).max()
        ),
        "max_gradient_identity_absolute_error": gradient_error,
        "duplicate_gram_max_difference": float((correct_gram - naive_gram).abs().max()),
        "depth_layer_probe": {
            "front_raw_rows": int(front_exported[6].numel()),
            "back_raw_rows": int(back_exported[6].numel()),
            "front_reduced_rows": int(front_pixels.numel()),
            "back_reduced_rows": int(back_pixels.numel()),
            "shared_reduced_keys": int(shared_depth_pixels.numel()),
            "front_shared_max_abs_feature": front_shared_max,
            "back_shared_max_abs_feature": back_shared_max,
            "full_minus_layer_sum_max_abs_error": depth_layer_sum_error,
        },
        "ordinary_parent_sha256": _sha256(parent_high),
        "exported_parent_sha256": _sha256(exported_parent_high),
        "active_render_sha256": _sha256(active_high),
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
        write_json_new(args.output, result)
    if result["decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
