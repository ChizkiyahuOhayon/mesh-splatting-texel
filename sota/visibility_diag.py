"""Zero-training premise check for SoftTail v2 on a frozen checkpoint.

For one trained model, accumulate the per-face dominant/hit counts over every
training view (no gradients, v1 representation settings: 4x supersampling,
cutoff 1e-4, no absorption), pool them to the surface-dominance ratio ``d_v``,
and test whether the vertices that actually sit at the trained opacity floor
are the low-dominance ones. Optionally, for a few fixed test views, write the
per-pixel dominance map next to the per-pixel error gain of this model over a
matched stock checkpoint and report their rank correlation.

Outputs ``diag.json`` plus PNGs and a ``DONE`` marker. It is a premise check
(is the signal there at all?), not a proxy for the trained effect size.
"""

import copy
import json
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from sota.visibility import face_visibility_counts, pool_to_vertices, roc_auc, spearman
from triangle_renderer import render
from utils.general_utils import safe_state


AT_FLOOR_MARGIN = 0.02
OPAQUE_ABOVE = 0.98


def _render(view, triangles, pipeline, background):
    return render(
        view, triangles, pipeline, background,
        transmittance_threshold_override=1e-4,
        absorb_transmittance_tail=False,
        upsample_override=4,
    )


def _load(dataset, iteration):
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(args=dataset, triangles=triangles, init_opacity=None, set_sigma=None,
                  load_iteration=iteration, shuffle=False)
    triangles.scaling = 4
    return triangles, scene


def _to_png(array, path, cmap="viridis", vmin=None, vmax=None):
    import matplotlib
    array = np.asarray(array, dtype=np.float32)
    mask = np.isfinite(array)
    lo = float(np.nanmin(array)) if vmin is None else vmin
    hi = float(np.nanmax(array)) if vmax is None else vmax
    scaled = np.zeros_like(array)
    if hi > lo:
        scaled[mask] = np.clip((array[mask] - lo) / (hi - lo), 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap](scaled)
    rgba[~mask] = (0.0, 0.0, 0.0, 1.0)
    Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)).save(path)


def _downsample(image_hw, factor):
    tensor = torch.as_tensor(image_hw)[None, None].float()
    return torch.nn.functional.avg_pool2d(tensor, factor)[0, 0]


def run(dataset, pipeline, args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda")
    pipeline.texel_footprint_filter = False

    triangles, scene = _load(dataset, args.iteration)
    faces = triangles.get_triangle_indices
    n_faces, n_vertices = faces.shape[0], triangles.get_vertices.shape[0]
    dominant = torch.zeros(n_faces, device="cuda")
    hits = torch.zeros(n_faces, device="cuda")
    train = sorted(scene.getTrainCameras(), key=lambda view: view.image_name)
    with torch.no_grad():
        for view in train:
            pkg = _render(view, triangles, pipeline, background)
            dom_f, hit_f = face_visibility_counts(pkg["rend_ids"], pkg["triangle_was_rendered"], n_faces)
            dominant += dom_f
            hits += hit_f
        dom_v, hit_v = pool_to_vertices(faces, dominant, hits, n_vertices)
        seen = hit_v > 0
        d_v = torch.ones(n_vertices, device="cuda")
        d_v[seen] = (dom_v[seen] / hit_v[seen]).clamp(0.0, 1.0)
        d_f = torch.ones(n_faces, device="cuda")
        d_f[hits > 0] = (dominant[hits > 0] / hits[hits > 0]).clamp(0.0, 1.0)

        realized = triangles.get_vertex_weight.reshape(-1)
        floor = (triangles.opacity_floor_vertex.reshape(-1)
                 if triangles.opacity_floor_vertex is not None
                 else torch.full_like(realized, float(triangles.opacity_floor)))
        at_floor = realized < floor + AT_FLOOR_MARGIN
        opaque = realized > OPAQUE_ABOVE
        labelled = at_floor | opaque
        auc = roc_auc(-d_v[labelled], at_floor[labelled])   # low dominance => at floor
        auc_seen = roc_auc(-d_v[labelled & seen], at_floor[labelled & seen])
        histogram = torch.histc(d_v[seen], bins=10, min=0.0, max=1.0)

        summary = {
            "experiment": "softtail-v2-visibility-premise-check-v1",
            "scene": args.scene,
            "model": str(Path(dataset.model_path).resolve()),
            "iteration": args.iteration,
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "training_views": len(train),
            "vertices": int(n_vertices),
            "faces": int(n_faces),
            "vertices_seen": int(seen.sum()),
            "checkpoint_opacity_floor": float(triangles.opacity_floor),
            "per_vertex_floor": triangles.opacity_floor_vertex is not None,
            "fraction_at_floor": float(at_floor.float().mean()),
            "fraction_opaque": float(opaque.float().mean()),
            "mean_d_at_floor": float(d_v[at_floor].mean()) if int(at_floor.sum()) else None,
            "mean_d_opaque": float(d_v[opaque].mean()) if int(opaque.sum()) else None,
            "auc_low_dominance_predicts_at_floor": auc,
            "auc_seen_only": auc_seen,
            "d_v_histogram_seen": [float(x) for x in histogram],
            "views": [],
        }
        torch.save({"d_v": d_v.cpu(), "d_f": d_f.cpu(), "realized": realized.cpu(), "floor": floor.cpu()},
                   output / "vertex_statistics.pt")

        stock = None
        if args.stock_model:
            stock_dataset = copy.copy(dataset)
            stock_dataset.model_path = args.stock_model
            stock, _ = _load(stock_dataset, args.iteration)
            stock.opacity_floor = 0.9999  # published endpoint of the matched baseline

        test = sorted(scene.getTestCameras(), key=lambda view: view.image_name)
        step = max(1, len(test) // max(1, args.views))
        for view in test[::step][:args.views]:
            pkg = _render(view, triangles, pipeline, background)
            ids = pkg["rend_ids"][0]
            dmap = torch.full_like(ids, float("nan"))
            valid = ids >= 0
            dmap[valid] = d_f[ids[valid].long()]
            dmap_small = _downsample(torch.nan_to_num(dmap, nan=1.0), 4).cpu().numpy()
            fg = (_downsample(valid.float(), 4) > 0.5).cpu().numpy()
            dmap_small[~fg] = np.nan
            _to_png(dmap_small, output / f"{view.image_name}_dominance.png", vmin=0.0, vmax=1.0)
            row = {"view": view.image_name, "foreground_pixels": int(fg.sum())}
            if stock is not None:
                target = view.original_image[:3].cuda().clamp(0.0, 1.0)
                ours = pkg["render"].clamp(0.0, 1.0)
                theirs = _render(view, stock, pipeline, background)["render"].clamp(0.0, 1.0)
                gain = ((theirs - target).abs().mean(0) - (ours - target).abs().mean(0)).cpu().numpy()
                _to_png(gain, output / f"{view.image_name}_gain.png", cmap="coolwarm",
                        vmin=-0.1, vmax=0.1)
                fg_t = torch.as_tensor(fg)
                row["spearman_dominance_vs_gain"] = spearman(
                    torch.as_tensor(dmap_small)[fg_t], torch.as_tensor(gain)[fg_t])
                row["mean_gain_low_dominance"] = float(np.mean(gain[fg & (dmap_small < 0.5)])) \
                    if np.any(fg & (dmap_small < 0.5)) else None
                row["mean_gain_high_dominance"] = float(np.mean(gain[fg & (dmap_small >= 0.5)])) \
                    if np.any(fg & (dmap_small >= 0.5)) else None
            summary["views"].append(row)

    with open(output / "diag.json", "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
        handle.write("\n")
    (output / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--stock_model", default="")
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    parsed = get_combined_args(parser)
    if not parsed.eval:
        parser.error("--eval is required")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
