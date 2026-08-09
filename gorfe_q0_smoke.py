"""GPU representation gate for GoRFE's native P2-DC/P2-SH1 carrier."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
from edgeval_core import p2_sh1_edge_radiance, vertex_sh1_factors
from utils.graphics_utils import getProjectionMatrix


PROBE_CHANNELS = torch.tensor([1.0, -0.3, 0.2])
COEFFICIENT_EPSILON = 1e-3
COEFFICIENT_RELATIVE_TOLERANCE = 5e-3
VERTEX_EPSILON = 2e-4
VERTEX_RELATIVE_TOLERANCE = 5e-2


def _sha256(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _rasterizer(device, camera_position):
    fov = 1.0
    view = torch.eye(4, dtype=torch.float32, device=device)
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).transpose(0, 1).to(device)
    settings = TriangleRasterizationSettings(
        image_height=64,
        image_width=64,
        tanfovx=math.tan(fov / 2),
        tanfovy=math.tan(fov / 2),
        bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=projection,
        sh_degree=0,
        campos=torch.tensor(camera_position, dtype=torch.float32, device=device),
        prefiltered=False,
        debug=False,
    )
    return TriangleRasterizer(settings)


def _default_vertices(device):
    return torch.tensor(
        [[-0.6, -0.5, 2.0], [0.6, -0.5, 2.0], [0.0, 0.6, 2.0]],
        dtype=torch.float32,
        device=device,
    )


def _render(rasterizer, edge_details=None, face_edge_ids=None, vertices_value=None):
    device = rasterizer.raster_settings.bg.device
    source_vertices = _default_vertices(device) if vertices_value is None else vertices_value
    vertices = source_vertices.detach().clone().requires_grad_(True)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    weights = torch.full((3,), 0.8, dtype=torch.float32, device=device, requires_grad=True)
    colors = torch.tensor(
        [[0.2, 0.3, 0.4], [0.4, 0.2, 0.1], [0.1, 0.5, 0.2]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    scaling = torch.zeros(1, dtype=torch.float32, device=device)
    output = rasterizer(
        vertices=vertices,
        triangles_indices=faces,
        vertex_weights=weights,
        sigma=0.0001,
        scaling=scaling,
        colors_precomp=colors,
        edge_details=edge_details,
        face_edge_ids=face_edge_ids,
    )
    return output[0], vertices


def _probe(image):
    channels = PROBE_CHANNELS.to(device=image.device)
    return (image * channels[:, None, None]).sum()


def _coefficient_gradient_check(rasterizer, detail, face_edge_ids):
    variable = detail.detach().clone().requires_grad_(True)
    rendered, vertices = _render(rasterizer, variable, face_edge_ids)
    analytic, vertex_gradient = torch.autograd.grad(_probe(rendered), (variable, vertices))
    finite_difference = torch.empty_like(variable)
    with torch.no_grad():
        for slot in range(4):
            for channel in range(3):
                plus = detail.detach().clone()
                minus = detail.detach().clone()
                plus[0, slot, channel] += COEFFICIENT_EPSILON
                minus[0, slot, channel] -= COEFFICIENT_EPSILON
                plus_probe = _probe(_render(rasterizer, plus, face_edge_ids)[0])
                minus_probe = _probe(_render(rasterizer, minus, face_edge_ids)[0])
                finite_difference[0, slot, channel] = (
                    plus_probe - minus_probe
                ) / (2.0 * COEFFICIENT_EPSILON)
    scale = torch.maximum(analytic.abs(), finite_difference.abs()).clamp_min(1.0)
    relative_error = (analytic - finite_difference).abs() / scale
    finite = bool(torch.isfinite(analytic).all() and torch.isfinite(finite_difference).all())
    return {
        "analytic": analytic,
        "finite_difference": finite_difference,
        "max_relative_error": float(relative_error.max()),
        "finite": finite,
        "matches": finite and float(relative_error.max()) <= COEFFICIENT_RELATIVE_TOLERANCE,
        "vertex_gradient": vertex_gradient,
    }


def _vertex_gradient_check(rasterizer, detail, face_edge_ids):
    variable = detail.detach().clone().requires_grad_(True)
    rendered, vertices = _render(rasterizer, variable, face_edge_ids)
    (analytic_vertices,) = torch.autograd.grad(_probe(rendered), vertices)
    coordinate = (0, 0)
    with torch.no_grad():
        plus = _default_vertices(vertices.device)
        minus = _default_vertices(vertices.device)
        plus[coordinate] += VERTEX_EPSILON
        minus[coordinate] -= VERTEX_EPSILON
        finite_difference = (
            _probe(_render(rasterizer, detail, face_edge_ids, plus)[0])
            - _probe(_render(rasterizer, detail, face_edge_ids, minus)[0])
        ) / (2.0 * VERTEX_EPSILON)
    analytic = analytic_vertices[coordinate]
    scale = torch.maximum(analytic.abs(), finite_difference.abs()).clamp_min(1.0)
    relative_error = float((analytic - finite_difference).abs() / scale)
    finite = bool(torch.isfinite(analytic) and torch.isfinite(finite_difference))
    return {
        "coordinate": list(coordinate),
        "analytic": float(analytic),
        "finite_difference": float(finite_difference),
        "relative_error": relative_error,
        "finite": finite,
        "matches": finite and relative_error <= VERTEX_RELATIVE_TOLERANCE,
    }


def _continuity_check():
    endpoints = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    camera = torch.tensor([0.2, -0.4, 0.1])
    left_factors = vertex_sh1_factors(
        torch.cat((endpoints, torch.tensor([[0.0, 1.0, 2.0]]))), camera
    )
    right_factors = vertex_sh1_factors(
        torch.cat((endpoints.flip(0), torch.tensor([[0.0, -1.0, 2.0]]))), camera
    )
    coefficients = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 11.0
    left = p2_sh1_edge_radiance(
        torch.tensor([0.25, 0.75, 0.0]), left_factors, coefficients, 0
    )
    right = p2_sh1_edge_radiance(
        torch.tensor([0.75, 0.25, 0.0]), right_factors, coefficients, 0
    )
    return bool(torch.equal(left, right)), left


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("GoRFE-Q0 requires CUDA")
    device = torch.device("cuda:0")
    face_edge_ids = torch.tensor([[0, -1, -1]], dtype=torch.int32, device=device)
    first_rasterizer = _rasterizer(device, [0.0, 0.0, 0.0])
    second_rasterizer = _rasterizer(device, [0.7, -0.2, 0.1])

    baseline = _render(first_rasterizer)[0]
    zero_detail = torch.zeros((1, 4, 3), dtype=torch.float32, device=device)
    zero_render = _render(first_rasterizer, zero_detail, face_edge_ids)[0]

    legacy_dc = torch.tensor([[0.125, -0.05, 0.075]], dtype=torch.float32, device=device)
    combined_dc = torch.zeros((1, 4, 3), dtype=torch.float32, device=device)
    combined_dc[:, 0, :] = legacy_dc
    legacy_dc_render = _render(first_rasterizer, legacy_dc, face_edge_ids)[0]
    combined_dc_render = _render(first_rasterizer, combined_dc, face_edge_ids)[0]

    detail = torch.tensor(
        [[[0.04, -0.03, 0.02], [0.12, -0.06, 0.03],
          [-0.07, 0.09, 0.05], [0.06, 0.02, -0.08]]],
        dtype=torch.float32,
        device=device,
    )
    first_render = _render(first_rasterizer, detail, face_edge_ids)[0]
    second_render = _render(second_rasterizer, detail, face_edge_ids)[0]
    first_gradient = _coefficient_gradient_check(first_rasterizer, detail, face_edge_ids)
    second_gradient = _coefficient_gradient_check(second_rasterizer, detail, face_edge_ids)
    vertex_gradient = _vertex_gradient_check(first_rasterizer, detail, face_edge_ids)
    continuous, boundary_value = _continuity_check()

    checks = {
        "zero_detail_is_bitwise_parent": torch.equal(baseline, zero_render),
        "legacy_dc_matches_combined_dc_bitwise": torch.equal(legacy_dc_render, combined_dc_render),
        "nonzero_angular_detail_changes_render": float((first_render - baseline).abs().max()) > 0.0,
        "camera_direction_changes_angular_render": float((first_render - second_render).abs().max()) > 0.0,
        "shared_edge_is_orientation_invariant_and_continuous": continuous,
        "first_camera_coefficient_gradient_is_finite": first_gradient["finite"],
        "first_camera_coefficient_gradient_matches_central_difference": first_gradient["matches"],
        "second_camera_coefficient_gradient_is_finite": second_gradient["finite"],
        "second_camera_coefficient_gradient_matches_central_difference": second_gradient["matches"],
        "vertex_gradient_is_finite": vertex_gradient["finite"],
        "vertex_gradient_matches_central_difference": vertex_gradient["matches"],
    }
    result = {
        "experiment": "GoRFE-Q0",
        "device": torch.cuda.get_device_name(0),
        "cuda_visible_device_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "checks": checks,
        "baseline_sha256": _sha256(baseline),
        "zero_render_sha256": _sha256(zero_render),
        "legacy_dc_render_sha256": _sha256(legacy_dc_render),
        "combined_dc_render_sha256": _sha256(combined_dc_render),
        "first_camera_render_sha256": _sha256(first_render),
        "second_camera_render_sha256": _sha256(second_render),
        "max_first_camera_coefficient_gradient_relative_error": first_gradient[
            "max_relative_error"
        ],
        "max_second_camera_coefficient_gradient_relative_error": second_gradient[
            "max_relative_error"
        ],
        "vertex_gradient_check": vertex_gradient,
        "first_camera_analytic_gradient": first_gradient["analytic"].detach().cpu().tolist(),
        "first_camera_finite_difference_gradient": first_gradient[
            "finite_difference"
        ].detach().cpu().tolist(),
        "second_camera_analytic_gradient": second_gradient["analytic"].detach().cpu().tolist(),
        "second_camera_finite_difference_gradient": second_gradient[
            "finite_difference"
        ].detach().cpu().tolist(),
        "shared_edge_boundary_value": boundary_value.tolist(),
        "max_first_camera_render_change": float((first_render - baseline).abs().max()),
        "max_camera_direction_render_change": float((first_render - second_render).abs().max()),
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if result["decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
