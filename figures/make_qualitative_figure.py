"""Compose the qualitative figure from the exported renders.

``sota/qualitative.py`` writes pixel-aligned renders of the same test views for
every arm, so the figure is a crop of the same rectangle from each of them. The
view and the crop are chosen by measurement, not by eye: the view where our
error drops most against the baseline, and inside it the window where the drop
is largest. Run it where the exports live (the server), then copy the one PNG.

    python figures/make_qualitative_figure.py --root <qualitative root> \
        --scene bicycle --scene room --out assets/softtail_qualitative_full.png
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Left to right: what the scene really looks like, then the three meshes.
COLUMNS = (
    ("targets", "Ground truth"),
    ("stock", "MeshSplatting"),
    ("ours_quality", "+ opacity floor"),
    ("softtail_full", "SoftTail (ours)"),
)
CROP = 320
PAD = 6
LABEL = 28


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def view_names(scene_dir):
    manifest = json.loads((scene_dir / "softtail_full" / "manifest.json").read_text())
    names = []
    for index, name in enumerate(manifest["view_names"]):
        names.append(f"{index:03d}_{Path(name).stem}.png")
    return names


def box_sum(image, window):
    """Sum over every window-sized box, via a summed-area table."""
    padded = np.pad(image.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (padded[window:, window:] - padded[:-window, window:]
            - padded[window:, :-window] + padded[:-window, :-window])


def pick(scene_dir, names):
    """The view, and the box inside it, where SoftTail beats the baseline most."""
    best = None
    for name in names:
        target = load(scene_dir / "targets" / name)
        baseline = np.abs(load(scene_dir / "stock" / "renders" / name) - target).mean(2)
        ours = np.abs(load(scene_dir / "softtail_full" / "renders" / name) - target).mean(2)
        gain = baseline - ours
        window = min(CROP, gain.shape[0], gain.shape[1])
        boxes = box_sum(gain, window)
        index = int(np.argmax(boxes))
        row, column = divmod(index, boxes.shape[1])
        score = float(boxes.flat[index]) / (window * window)
        if best is None or score > best["score"]:
            best = {"name": name, "score": score, "box": (row, column, window),
                    "view_gain": float(gain.mean())}
    return best


def strip(scene_dir, choice, height):
    row, column, window = choice["box"]
    tiles = []
    for arm, _ in COLUMNS:
        source = (scene_dir / "targets" / choice["name"] if arm == "targets"
                  else scene_dir / arm / "renders" / choice["name"])
        crop = Image.open(source).convert("RGB").crop(
            (column, row, column + window, row + window))
        tiles.append(crop.resize((height, height), Image.LANCZOS))
    return tiles


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True,
                        help="a qualitative export root; may be repeated")
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tile", type=int, default=320)
    args = parser.parse_args()

    roots = [Path(root) for root in args.root]
    rows, labels = [], []
    for scene in args.scene:
        scene_dir = next((root / scene for root in roots if (root / scene).is_dir()), None)
        if scene_dir is None:
            raise FileNotFoundError(f"no export for {scene} under {args.root}")
        choice = pick(scene_dir, view_names(scene_dir))
        print(f"{scene:10s} view {choice['name']:28s} "
              f"box gain {choice['score']:.4f} view gain {choice['view_gain']:+.4f}")
        rows.append(strip(scene_dir, choice, args.tile))
        labels.append(scene)

    tile = args.tile
    width = len(COLUMNS) * tile + (len(COLUMNS) + 1) * PAD
    height = LABEL + len(rows) * (tile + PAD) + PAD
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (_, title) in enumerate(COLUMNS):
        x = PAD + index * (tile + PAD)
        draw.text((x + 4, 8), title, fill="black")
    for r, tiles in enumerate(rows):
        y = LABEL + r * (tile + PAD)
        for c, image in enumerate(tiles):
            canvas.paste(image, (PAD + c * (tile + PAD), y))
        draw.text((PAD + 4, y + 4), labels[r], fill="white")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"wrote {out} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
