"""Narrow scene boundary for GoRFE-V1 without constructing ``Scene``.

Camera geometry is built from COLMAP metadata alone.  Target pixels enter only
through :func:`load_training_rgb`, after the official-split guard has accepted
the immutable camera identity.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from gorfe_v1_prepare_core import (
    MetadataCamera,
    OfficialSplitGuard,
    validate_checkpoint_state,
)
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2


DEFAULT_ZNEAR = 0.01
DEFAULT_ZFAR = 100.0
DEFAULT_MAX_WIDTH = 1600
LOCKED_SCALING = 4


class TargetFreeMiniCam:
    """Renderer-compatible camera carrying geometry and no target tensor."""

    def __init__(
        self,
        *,
        metadata: MetadataCamera,
        uid: int,
        width: int,
        height: int,
        fovx: float,
        fovy: float,
        world_view_transform: torch.Tensor,
        full_proj_transform: torch.Tensor,
    ):
        self.uid = uid
        # ``Camera(loadCam(...))`` receives CameraInfo.uid, which the COLMAP
        # reader sets to the intrinsic camera id rather than the image id.
        self.colmap_id = metadata.camera_id
        self.image_name = metadata.image_name
        self.relative_image_path = metadata.relative_image_path
        self.R = _qvec2rotmat(metadata.qvec).transpose()
        self.T = np.asarray(metadata.tvec, dtype=np.float64)
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy
        self.znear = DEFAULT_ZNEAR
        self.zfar = DEFAULT_ZFAR
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = torch.inverse(world_view_transform)[3, :3]


def _qvec2rotmat(qvec) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("COLMAP qvec must contain four finite values")
    return np.array(
        [
            [
                1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
                2 * q[1] * q[2] - 2 * q[0] * q[3],
                2 * q[3] * q[1] + 2 * q[0] * q[2],
            ],
            [
                2 * q[1] * q[2] + 2 * q[0] * q[3],
                1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
                2 * q[2] * q[3] - 2 * q[0] * q[1],
            ],
            [
                2 * q[3] * q[1] - 2 * q[0] * q[2],
                2 * q[2] * q[3] + 2 * q[0] * q[1],
                1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
            ],
        ]
    )


def resolution_minus_one_size(
    original_width: int, original_height: int
) -> tuple[int, int]:
    """Exact ``loadCam(..., resolution=-1, resolution_scale=1)`` dimensions."""
    if (
        not isinstance(original_width, int)
        or isinstance(original_width, bool)
        or not isinstance(original_height, int)
        or isinstance(original_height, bool)
        or original_width < 1
        or original_height < 1
    ):
        raise ValueError("original image dimensions must be positive integers")
    global_down = original_width / DEFAULT_MAX_WIDTH if original_width > DEFAULT_MAX_WIDTH else 1.0
    width = int(original_width / global_down)
    height = int(original_height / global_down)
    if width < 1 or height < 1:
        raise ValueError("resolution=-1 produced an empty image")
    return width, height


def _camera_fovs(metadata: MetadataCamera) -> tuple[float, float]:
    if metadata.model == "SIMPLE_PINHOLE" and len(metadata.parameters) == 3:
        focal_x = focal_y = metadata.parameters[0]
    elif metadata.model == "PINHOLE" and len(metadata.parameters) == 4:
        focal_x, focal_y = metadata.parameters[:2]
    else:
        raise ValueError(
            "GoRFE-V1 cameras must use SIMPLE_PINHOLE(3) or PINHOLE(4) metadata"
        )
    if not math.isfinite(focal_x) or not math.isfinite(focal_y) or focal_x <= 0 or focal_y <= 0:
        raise ValueError("camera focal lengths must be finite and positive")
    return focal2fov(focal_x, metadata.width), focal2fov(focal_y, metadata.height)


def make_target_free_minicam(
    metadata: MetadataCamera,
    *,
    uid: int,
    device: torch.device | str = "cuda",
) -> TargetFreeMiniCam:
    """Build the same transforms as ``Camera`` without opening its image."""
    if not isinstance(metadata, MetadataCamera):
        raise TypeError("metadata must be a MetadataCamera")
    if not isinstance(uid, int) or isinstance(uid, bool) or uid < 0:
        raise ValueError("uid must be a nonnegative integer")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("camera device must be cpu or cuda")
    width, height = resolution_minus_one_size(metadata.width, metadata.height)
    fovx, fovy = _camera_fovs(metadata)
    rotation = _qvec2rotmat(metadata.qvec).transpose()
    translation = np.asarray(metadata.tvec, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("COLMAP tvec must contain three finite values")

    world_view = torch.tensor(getWorld2View2(rotation, translation)).transpose(0, 1).to(device)
    projection = getProjectionMatrix(
        znear=DEFAULT_ZNEAR,
        zfar=DEFAULT_ZFAR,
        fovX=fovx,
        fovY=fovy,
    ).transpose(0, 1).to(device)
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return TargetFreeMiniCam(
        metadata=metadata,
        uid=uid,
        width=width,
        height=height,
        fovx=fovx,
        fovy=fovy,
        world_view_transform=world_view,
        full_proj_transform=full_projection,
    )


def load_training_rgb(
    metadata: MetadataCamera,
    *,
    split_guard: OfficialSplitGuard,
    image_root: os.PathLike[str] | str,
) -> torch.Tensor:
    """Decode one accepted training target with the published resize operation."""
    if not isinstance(metadata, MetadataCamera):
        raise TypeError("metadata must be a MetadataCamera")
    # This call deliberately precedes path resolution, existence checks, PIL
    # import, and decode.  An official test identity cannot cause image I/O.
    split_guard.require_training(metadata.image_name)

    from PIL import Image

    image_path = Path(image_root).expanduser().resolve() / Path(
        metadata.relative_image_path
    ).name
    width, height = resolution_minus_one_size(metadata.width, metadata.height)
    with Image.open(image_path) as image:
        if image.size != (metadata.width, metadata.height):
            raise ValueError(
                f"image dimensions drifted for {metadata.image_name}: "
                f"{image.size} != {(metadata.width, metadata.height)}"
            )
        if len(image.split()) > 3:
            rgb = torch.cat(
                [PILtoTorch(channel, (width, height)) for channel in image.split()[:3]],
                dim=0,
            )
        else:
            rgb = PILtoTorch(image, (width, height))
    if rgb.shape != (3, height, width) or rgb.dtype != torch.float32:
        raise ValueError(
            f"training target {metadata.image_name} is not RGB after resize: "
            f"shape={tuple(rgb.shape)}, dtype={rgb.dtype}"
        )
    if not bool(torch.isfinite(rgb).all()) or bool((rgb < 0).any()) or bool((rgb > 1).any()):
        raise ValueError(f"training target {metadata.image_name} has invalid RGB values")
    return rgb.contiguous()


def _frozen_copy(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return value.detach().to(device=device, dtype=dtype, copy=True).contiguous()


def load_frozen_triangle_model(
    state: Mapping[str, Any],
    *,
    device: torch.device | str = "cuda",
):
    """Construct only the frozen tensors needed by ``triangle_renderer.render``.

    Unlike ``Scene``/``TriangleModel.load_parameters``, this function creates no
    optimizer and allocates no per-face training buffers.
    """
    details = validate_checkpoint_state(state)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("triangle model device must be cpu or cuda")

    # Kept lazy so metadata-only helpers and their CPU tests do not import the
    # Scene package or any CUDA-only triangulation dependency.
    from scene.triangle_model import TriangleModel

    model = TriangleModel(details["active_sh_degree"])
    model.vertices = _frozen_copy(state["triangles_points"], device=device, dtype=torch.float32)
    model._triangle_indices = _frozen_copy(
        state["_triangle_indices"], device=device, dtype=torch.int32
    )
    model.vertex_weight = _frozen_copy(state["vertex_weight"], device=device, dtype=torch.float32)
    model._features_dc = _frozen_copy(state["features_dc"], device=device, dtype=torch.float32)
    model._features_rest = _frozen_copy(
        state["features_rest"], device=device, dtype=torch.float32
    )
    model._triangles = torch.empty(0, dtype=torch.float32, device=device)
    raw_sigma = state["sigma"]
    if torch.is_tensor(raw_sigma):
        raw_sigma = raw_sigma.detach().cpu().item()
    model._sigma = float(raw_sigma)
    model.active_sh_degree = details["active_sh_degree"]
    model.opacity_floor = 0.999
    model.texel_order = 0
    model._texels = torch.empty(0, dtype=torch.float32, device=device)
    model.scaling = LOCKED_SCALING

    # Preserve the public attribute contract without allocating F-sized training
    # state or optimizer moment tensors.
    model.optimizer = None
    model.texel_optimizer = None
    model.image_size = 0
    model.importance_score = 0
    model.pixel_count = 0
    for tensor in (
        model.vertices,
        model.vertex_weight,
        model._features_dc,
        model._features_rest,
    ):
        tensor.requires_grad_(False)
    return model
