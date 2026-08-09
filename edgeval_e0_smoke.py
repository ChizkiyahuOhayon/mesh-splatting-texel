"""GPU smoke gate for the native EdgeVal P2 edge-color carrier."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
from utils.graphics_utils import getProjectionMatrix


def _sha256(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _rasterizer(device):
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
        campos=torch.zeros(3, dtype=torch.float32, device=device),
        prefiltered=False,
        debug=False,
    )
    return TriangleRasterizer(settings)


def _render(rasterizer, edge_details=None, face_edge_ids=None):
    device = rasterizer.raster_settings.bg.device
    vertices = torch.tensor(
        [[-0.6, -0.5, 2.0], [0.6, -0.5, 2.0], [0.0, 0.6, 2.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
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
    return output[0]


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("EdgeVal-E0 requires CUDA")
    device = torch.device("cuda:0")
    rasterizer = _rasterizer(device)

    baseline = _render(rasterizer)
    zero_detail = torch.zeros((1, 3), dtype=torch.float32, device=device, requires_grad=True)
    face_edge_ids = torch.tensor([[0, -1, -1]], dtype=torch.int32, device=device)
    zero_render = _render(rasterizer, zero_detail, face_edge_ids)
    bitwise_zero_identity = torch.equal(baseline, zero_render)

    detail = torch.tensor(
        [[0.125, -0.05, 0.075]], dtype=torch.float32, device=device, requires_grad=True
    )
    rendered = _render(rasterizer, detail, face_edge_ids)
    probe = (rendered * torch.tensor([1.0, -0.3, 0.2], device=device)[:, None, None]).sum()
    (analytic_gradient,) = torch.autograd.grad(probe, detail)

    epsilon = 1e-3
    finite_difference = torch.empty_like(detail)
    with torch.no_grad():
        for channel in range(3):
            plus = detail.detach().clone()
            minus = detail.detach().clone()
            plus[0, channel] += epsilon
            minus[0, channel] -= epsilon
            plus_probe = (
                _render(rasterizer, plus, face_edge_ids)
                * torch.tensor([1.0, -0.3, 0.2], device=device)[:, None, None]
            ).sum()
            minus_probe = (
                _render(rasterizer, minus, face_edge_ids)
                * torch.tensor([1.0, -0.3, 0.2], device=device)[:, None, None]
            ).sum()
            finite_difference[0, channel] = (plus_probe - minus_probe) / (2.0 * epsilon)

    gradient_abs_error = (analytic_gradient - finite_difference).abs()
    gradient_scale = torch.maximum(analytic_gradient.abs(), finite_difference.abs()).clamp_min(1.0)
    gradient_rel_error = gradient_abs_error / gradient_scale
    finite_gradient = bool(torch.isfinite(analytic_gradient).all() and torch.isfinite(finite_difference).all())
    gradient_matches = finite_gradient and float(gradient_rel_error.max()) <= 5e-3
    nonzero_effect = float((rendered - baseline).abs().max()) > 0.0

    result = {
        "experiment": "EdgeVal-E0",
        "device": torch.cuda.get_device_name(0),
        "cuda_visible_device_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "checks": {
            "zero_detail_is_bitwise_parent": bitwise_zero_identity,
            "nonzero_detail_changes_render": nonzero_effect,
            "edge_gradient_is_finite": finite_gradient,
            "edge_gradient_matches_central_difference": gradient_matches,
        },
        "baseline_sha256": _sha256(baseline),
        "zero_render_sha256": _sha256(zero_render),
        "nonzero_render_sha256": _sha256(rendered),
        "analytic_gradient": analytic_gradient.detach().cpu().tolist(),
        "finite_difference_gradient": finite_difference.detach().cpu().tolist(),
        "max_gradient_relative_error": float(gradient_rel_error.max()),
        "max_render_change": float((rendered - baseline).abs().max()),
    }
    result["decision"] = "pass" if all(result["checks"].values()) else "fail"
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
