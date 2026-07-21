#!/usr/bin/env python3
"""
Export a trained mesh-splatting checkpoint to a compact .msplat binary
for the WebGL viewer. The spherical harmonics are compressed here: the DC
term is baked into an 8-bit per-vertex color and levels 1-3 are quantized
per level and bit-packed into integer textures.

Binary layout (little-endian)
─────────────
  "MSPLAT\n"                         7 bytes   magic
  num_vertices     u32 LE            4 bytes
  num_triangles    u32 LE            4 bytes
  sh_degree        u32 LE            4 bytes
  sh1/2/3_max      f32×3 LE         12 bytes   per-level max(|coef|), 0 if absent
  camera_center    f32×3 LE         12 bytes
  camera_up        f32×3 LE         12 bytes
  camera_distance  f32 LE            4 bytes
  positions        f32×3 × V        12·V bytes
  indices          u32×3 × T        12·T bytes
  colors           u8×3  × V         3·V bytes   DC baked to RGB
  sh1_packed       u32×2 × P                     if sh_degree >= 1
  sh2_packed       u32×4 × P                     if sh_degree >= 2
  sh3_packed       u32×4 × P                     if sh_degree >= 3

SH textures are 2048 texels wide and indexed by vertex, so P = 2048·ceil(V/2048)
(last row zero-padded). Levels pack 9/15/21 signed coefficients at 7/8/6 bits; the
viewer decodes with `quantized · shK_max / qrange` (qrange = 63/127/31).

Usage
─────
  python export_web.py <model_dir> [--out FILE] [--degree D]

<model_dir> must contain point_cloud/iteration_XXXXX/point_cloud_state_dict.pt
and cameras.json.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch


def find_checkpoint(model_dir: str) -> str:
    """Return path to the latest point_cloud_state_dict.pt inside model_dir."""
    pc_root = os.path.join(model_dir, "point_cloud")
    if not os.path.isdir(pc_root):
        # Maybe the user pointed directly at the iteration folder
        candidate = os.path.join(model_dir, "point_cloud_state_dict.pt")
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            f"No point_cloud/ directory or point_cloud_state_dict.pt in '{model_dir}'"
        )
    iterations = sorted(
        [d for d in os.listdir(pc_root) if d.startswith("iteration_")],
        key=lambda d: int(d.split("_")[1]),
    )
    if not iterations:
        raise FileNotFoundError(f"No iteration_* folders inside '{pc_root}'")
    latest = iterations[-1]
    path = os.path.join(pc_root, latest, "point_cloud_state_dict.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing checkpoint at '{path}'")
    return path


def load_cameras_json(model_dir: str):
    """Return list of camera dicts from cameras.json."""
    path = os.path.join(model_dir, "cameras.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_camera_params(cameras, mesh_center):
    """
    Derive orbit-camera parameters from the training cameras.

    Returns (center, up, distance) where
      center   = mesh bounding-box centroid  (passed in)
      up       = average camera up direction (negative of the Y column of
                 each COLMAP rotation, since COLMAP Y points down in image)
      distance = median distance from camera positions to center
    """
    positions = []
    up_accum = np.zeros(3, dtype=np.float64)

    for cam in cameras:
        pos = np.array(cam["position"], dtype=np.float64)
        positions.append(pos)
        R = np.array(cam["rotation"], dtype=np.float64)  # 3×3 row-major
        # In COLMAP convention the camera Y axis points downward in the image,
        # so the "up" direction in world space is -R[:,1]  (negative second column).
        cam_up = -R[:, 1]
        up_accum += cam_up / np.linalg.norm(cam_up)

    up = up_accum / np.linalg.norm(up_accum)
    positions = np.stack(positions)
    dists = np.linalg.norm(positions - mesh_center[None, :], axis=1)
    distance = float(np.median(dists))
    return mesh_center.astype(np.float32), up.astype(np.float32), np.float32(distance)


def main():
    parser = argparse.ArgumentParser(
        description="Export mesh-splatting checkpoint to .msplat for the WebGL viewer."
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Model directory (contains point_cloud/ and cameras.json)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .msplat path (default: <model_dir>/scene.msplat)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=None,
        help="SH degree to export (default: model's active_sh_degree)",
    )
    args = parser.parse_args()

    # --- locate and load checkpoint ---
    ckpt_path = find_checkpoint(args.model_dir)
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    verts = state["triangles_points"].detach().float()  # [V, 3]
    faces = state["_triangle_indices"].detach().int()  # [T, 3]
    f_dc = state["features_dc"].detach().float()  # [V, 1, 3]
    f_rest = state["features_rest"].detach().float()  # [V, K, 3]
    model_deg = int(state.get("active_sh_degree", 3))

    export_deg = args.degree if args.degree is not None else model_deg
    if export_deg > model_deg:
        print(
            f"Warning: requested degree {export_deg} > model degree {model_deg}; "
            f"clamping to {model_deg}"
        )
        export_deg = model_deg

    V = verts.shape[0]
    T = faces.shape[0]
    num_coeffs = (export_deg + 1) ** 2

    print(f"Vertices: {V},  Triangles: {T}")
    print(f"Model SH degree: {model_deg},  Export SH degree: {export_deg}")
    print(f"SH coefficients per channel: {num_coeffs}")

    # --- assemble SH data  [V, num_coeffs, 3] ---
    # f_dc is [V, 1, 3], f_rest is [V, model_coeffs-1, 3]
    all_sh = torch.cat([f_dc, f_rest], dim=1)  # [V, model_coeffs, 3]
    all_sh = all_sh[:, :num_coeffs, :]  # truncate if needed
    sh = all_sh.numpy().astype(np.float32)  # coefficient-major, DC first

    # DC (level 0) -> 8-bit per-vertex color: round(clamp(0.5 + SH_C0·dc, 0, 1)·255).
    SH_C0 = 0.28209479177387814
    colors = np.round(np.clip(0.5 + SH_C0 * sh[:, 0, :], 0.0, 1.0) * 255.0).astype(np.uint8)

    # Levels 1-3 -> per-level quantized, bit-packed integer textures (padded 2048 wide).
    # level: (coeff count, signed range, bits/value, packed words, base coeff index)
    SH_LEVELS = {1: (3, 63, 7, 2, 1), 2: (5, 127, 8, 4, 4), 3: (7, 31, 6, 4, 9)}
    texels = 2048 * ((V + 2047) // 2048)
    sh_max = [0.0, 0.0, 0.0]
    sh_packed = []
    for lvl in range(1, export_deg + 1):
        count, qrange, bits, num_words, base = SH_LEVELS[lvl]
        vals = sh[:, base:base + count, :].reshape(V, -1)  # c0.r c0.g c0.b c1.r ...
        level_max = float(np.abs(vals).max())
        sh_max[lvl - 1] = level_max
        q = np.clip(np.round(vals * (qrange / (level_max or 1.0))), -qrange, qrange).astype(np.int32)
        masked = (q & ((1 << bits) - 1)).astype(np.uint64)  # two's-complement low bits
        words = np.zeros((texels, num_words), dtype=np.uint64)
        for i in range(vals.shape[1]):
            w, off = (i * bits) // 32, (i * bits) % 32
            words[:V, w] |= (masked[:, i] << off) & 0xFFFFFFFF
            if off + bits > 32:
                words[:V, w + 1] |= masked[:, i] >> (32 - off)
        sh_packed.append(words.astype(np.uint32))
        print(f"  level {lvl}: max(|coef|)={level_max:.4f}")

    # --- camera parameters ---
    mesh_center = verts.numpy().mean(axis=0).astype(np.float64)
    cameras = load_cameras_json(args.model_dir)
    if cameras is not None:
        cam_center, cam_up, cam_dist = compute_camera_params(cameras, mesh_center)
        print(f"Camera center: {cam_center}")
        print(f"Camera up:     {cam_up}")
        print(f"Camera dist:   {cam_dist}")
    else:
        print("Warning: cameras.json not found; using default camera params")
        cam_center = mesh_center.astype(np.float32)
        cam_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        cam_dist = np.float32(5.0)

    # --- write binary ---
    out_path = args.out or os.path.join(args.model_dir, "scene.msplat")
    print(f"Writing {out_path} ...")

    with open(out_path, "wb") as f:
        # magic
        f.write(b"MSPLAT\n")
        # header
        f.write(struct.pack("<I", V))
        f.write(struct.pack("<I", T))
        f.write(struct.pack("<I", export_deg))
        f.write(struct.pack("<3f", *sh_max))
        f.write(struct.pack("<3f", *cam_center.tolist()))
        f.write(struct.pack("<3f", *cam_up.tolist()))
        f.write(struct.pack("<f", float(cam_dist)))
        # positions  f32×3×V
        f.write(verts.numpy().astype(np.float32).tobytes())
        # indices    u32×3×T
        f.write(faces.numpy().astype(np.uint32).tobytes())
        # colors     u8×3×V
        f.write(colors.tobytes())
        # packed SH  u32 textures, levels 1..export_deg
        for words in sh_packed:
            f.write(words.tobytes())

    file_size = os.path.getsize(out_path)
    print(f"Done. File size: {file_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
