"""Native opaque-mesh renderer used by the preregistered FMMS G0 gate."""

import torch
import torch.nn.functional as F

from utils.sh_utils import eval_sh


class NativeRasterContext:
    """Hold the CUDA rasterizer and the fixed-mesh topology hash across views."""

    def __init__(self, dr):
        self.dr = dr
        self.raster = dr.RasterizeCudaContext()
        self.topology_hash = None

    def get_topology_hash(self, faces):
        if self.topology_hash is None:
            self.topology_hash = self.dr.antialias_construct_topology_hash(faces)
        return self.topology_hash


def vertex_clip_positions(vertices, full_proj_transform):
    """Match MeshSplatting's row-vector projection convention."""
    ones = torch.ones_like(vertices[:, :1])
    vertices_h = torch.cat((vertices, ones), dim=1)
    return vertices_h @ full_proj_transform


def vertex_sh_colors(triangles, camera):
    """Evaluate the same per-vertex SH colors as the baseline CUDA renderer."""
    sh = triangles.get_features.transpose(1, 2).contiguous()
    directions = triangles.get_vertices - camera.camera_center[None]
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.clamp_min(eval_sh(triangles.active_sh_degree, sh, directions) + 0.5, 0.0)


def create_raster_context():
    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise RuntimeError(
            "FMMS G0 requires nvdiffrast; install requirements-g0.txt first."
        ) from exc
    return NativeRasterContext(dr)


def render_native(camera, triangles, background, context, scale=1, antialias=True):
    """Render an opaque connected mesh with a hard z-buffer and optional analytic AA."""
    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise RuntimeError(
            "FMMS G0 requires nvdiffrast; install requirements-g0.txt first."
        ) from exc

    if scale not in (1, 2):
        raise ValueError(f"G0 native scale must be 1 or 2, got {scale}.")
    if getattr(triangles, "texel_order", 0) != 0:
        raise ValueError("G0 accepts baseline checkpoints only (texel_order must be 0).")

    height = int(camera.image_height) * scale
    width = int(camera.image_width) * scale
    vertices = triangles.get_vertices
    faces = triangles.get_triangle_indices.to(torch.int32).contiguous()
    pos_clip = vertex_clip_positions(vertices, camera.full_proj_transform)[None].contiguous()
    colors = vertex_sh_colors(triangles, camera)[None].contiguous()

    rast, _ = dr.rasterize(context.raster, pos_clip, faces, resolution=[height, width])
    foreground, _ = dr.interpolate(colors, rast, faces)
    alpha = (rast[..., 3:4] > 0).to(foreground.dtype)
    bg = background.reshape(1, 1, 1, 3).to(foreground)
    image = foreground * alpha + bg * (1.0 - alpha)

    if antialias:
        image = dr.antialias(
            image, rast, pos_clip, faces,
            topology_hash=context.get_topology_hash(faces),
        )

    image = image.permute(0, 3, 1, 2)
    alpha = alpha.permute(0, 3, 1, 2)
    if scale > 1:
        output_size = (int(camera.image_height), int(camera.image_width))
        image = F.interpolate(image, size=output_size, mode="area")
        alpha = F.interpolate(alpha, size=output_size, mode="area")

    return {"render": image[0], "rend_alpha": alpha[0]}
