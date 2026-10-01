"""Save the faces segmentation/segment.py assigned to one object as a new model.

    python -m sota.extract_object -m <run> --out <dir> [--ratio_threshold 0.75]

The output directory is a regular model (cfg_args, cameras.json,
point_cloud/iteration_30000) that every renderer and create_ply.py accept.
"""

import shutil
from argparse import ArgumentParser
from pathlib import Path

from arguments import ModelParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ratio_threshold", default=0.75, type=float)
    parser.add_argument("--out", required=True)
    args = get_combined_args(parser)
    dataset = model.extract(args)
    triangles = TriangleModel(dataset.sh_degree)
    Scene(args=dataset, triangles=triangles, init_opacity=None, set_sigma=None,
          load_iteration=args.iteration, shuffle=False, segment=True,
          ratio_threshold=args.ratio_threshold)
    out = Path(args.out)
    (out / "point_cloud").mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "cameras.json"):
        shutil.copyfile(Path(dataset.model_path) / name, out / name)
    triangles.save_parameters(str(out / "point_cloud" / "iteration_30000"))
    print("object faces:", int(triangles.get_triangle_indices.shape[0]))
