"""Cull a DTU mesh with the public 2DGS foreground-mask protocol.

The projection and dilation procedure follows
https://github.com/hbb1/2d-gaussian-splatting/tree/main/scripts/eval_dtu.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from skimage.morphology import binary_dilation, disk
from tqdm import tqdm

from render_utils import load_K_Rt_from_P


def cull_mesh(input_mesh, scan_dir, output_mesh):
    scan_dir = Path(scan_dir)
    image_paths = sorted((scan_dir / "images").glob("*.png"))
    mask_paths = sorted((scan_dir / "mask").glob("*.png"))
    if not image_paths or len(mask_paths) != len(image_paths):
        raise ValueError(
            f"expected one PNG mask per image in {scan_dir}; "
            f"found {len(image_paths)} images and {len(mask_paths)} masks"
        )

    camera_dict = np.load(scan_dir / "cameras.npz")
    scale_mats = [
        camera_dict[f"scale_mat_{index}"].astype(np.float32)
        for index in range(len(image_paths))
    ]
    world_mats = [
        camera_dict[f"world_mat_{index}"].astype(np.float32)
        for index in range(len(image_paths))
    ]

    mesh = trimesh.load_mesh(input_mesh, process=False)
    vertices = torch.as_tensor(
        np.asarray(mesh.vertices), dtype=torch.float32, device="cuda"
    )
    vertices = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1).T
    keep = torch.ones(vertices.shape[1], dtype=torch.bool, device="cuda")

    for world_mat, scale_mat, mask_path in tqdm(
        zip(world_mats, scale_mats, mask_paths),
        total=len(mask_paths),
        desc="Culling mesh with DTU masks",
    ):
        projection = (world_mat @ scale_mat)[:3, :4]
        intrinsic, pose = load_K_Rt_from_P(None, projection)
        intrinsic = torch.as_tensor(intrinsic, dtype=torch.float32, device="cuda")
        world_to_camera = torch.linalg.inv(
            torch.as_tensor(pose, dtype=torch.float32, device="cuda")
        )

        camera_points = intrinsic @ world_to_camera @ vertices
        pixel = (camera_points[:2] / (camera_points[2:3] + 1e-6)).T

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"could not read {mask_path}")
        height, width = mask.shape
        pixel[:, 0] /= width - 1
        pixel[:, 1] /= height - 1
        pixel = (pixel - 0.5) * 2
        valid = ((pixel > -1.0) & (pixel < 1.0)).all(dim=1)

        dilated = binary_dilation(mask > 0, disk(24)).astype(np.float32)
        mask_tensor = torch.as_tensor(dilated, device="cuda")[None, None]
        sampled = F.grid_sample(
            mask_tensor,
            pixel[None, None],
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0, 0]
        keep &= (sampled > 0) | ~valid

    face_keep = keep[torch.as_tensor(mesh.faces, device="cuda")].all(dim=1)
    mesh.update_faces(face_keep.cpu().numpy())
    mesh.remove_unreferenced_vertices()

    scale_mat = scale_mats[0]
    mesh.vertices = (
        np.asarray(mesh.vertices) * scale_mat[0, 0] + scale_mat[:3, 3][None]
    )
    output_mesh = Path(output_mesh)
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_mesh)
    print(
        f"Saved {len(mesh.vertices)} vertices and {len(mesh.faces)} faces "
        f"to {output_mesh}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mesh", required=True)
    parser.add_argument("--scan-dir", required=True)
    parser.add_argument("--output-mesh", required=True)
    args = parser.parse_args()
    cull_mesh(args.input_mesh, args.scan_dir, args.output_mesh)


if __name__ == "__main__":
    main()
