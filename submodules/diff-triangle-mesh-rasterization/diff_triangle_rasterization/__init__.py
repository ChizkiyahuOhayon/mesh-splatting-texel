#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_triangles(
    vertices,
    triangles_indices,
    vertex_weights,
    sigma,
    sh,
    colors_precomp,
    scaling,
    raster_settings,
    texels=None,
    edge_details=None,
    face_edge_ids=None,
    window_donors=None,
    sigma_face=None,
):
    return _RasterizeTriangles.apply(
        vertices,
        triangles_indices,
        vertex_weights,
        sigma,
        sh,
        colors_precomp,
        scaling,
        raster_settings,
        texels,
        edge_details,
        face_edge_ids,
        window_donors,
        sigma_face,
    )

_GORFE_DIAGNOSTIC_NAMES = (
    "raw_rows",
    "count_alpha_accepted_fragments",
    "count_blended_fragments",
    "write_alpha_accepted_fragments",
    "write_blended_fragments",
    "replay_transmittance_mismatch_pixels",
    "replay_last_contributor_mismatch_pixels",
    "count_write_mismatch_pixels",
    "write_overflow_rows",
    "high_resolution_pixels",
    "output_pixels",
    "replay_passes",
)


def export_gorfe_rows(
    vertices,
    triangles_indices,
    sigma,
    gorfe_face_edge_ids,
    gorfe_edge_count,
    image_height,
    image_width,
    output_height,
    output_width,
    output_scaling,
    campos,
    geom_buffer,
    num_rendered,
    binning_buffer,
    image_buffer,
    debug=False,
):
    """Replay one completed forward pass and export its exact sparse design.

    This low-level API intentionally requires the three opaque buffers and the
    per-face acceptance counts returned by that same native forward call.  It
    never invokes preprocessing or sorting again, and never changes renderer
    state.
    """
    # Match the ordinary renderer boundary, which reads campos through a
    # contiguous copy.  Real camera centers are row slices of an inverse 4x4
    # transform and therefore need not already have contiguous storage.
    campos = campos.contiguous()
    pixel_ids, group_ids, features, diagnostic_tensor = _C.export_gorfe_rows(
        vertices,
        triangles_indices,
        sigma,
        gorfe_face_edge_ids,
        gorfe_edge_count,
        image_height,
        image_width,
        output_height,
        output_width,
        output_scaling,
        campos,
        geom_buffer,
        num_rendered,
        binning_buffer,
        image_buffer,
        debug,
    )
    values = diagnostic_tensor.detach().cpu().tolist()
    if len(values) != len(_GORFE_DIAGNOSTIC_NAMES):
        raise RuntimeError(
            "native GoRFE diagnostics have an unexpected length: "
            f"{len(values)} != {len(_GORFE_DIAGNOSTIC_NAMES)}"
        )
    diagnostics = dict(zip(_GORFE_DIAGNOSTIC_NAMES, map(int, values)))
    return pixel_ids, group_ids, features, diagnostics


class _RasterizeTriangles(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vertices,
        triangles_indices,
        vertex_weights,
        sigma,
        sh,
        colors_precomp,
        scaling,
        raster_settings,
        texels,
        edge_details,
        face_edge_ids,
        window_donors,
        sigma_face,
    ):
        # texel_order 0 disables the carrier entirely: the C++ side then receives a
        # null pointer and the kernels take exactly the original code path, so the
        # unmodified baseline remains bit-reproducible.
        texel_order = getattr(raster_settings, "texel_order", 0) or 0
        if texels is None:
            texels = torch.zeros(0, device=vertices.device, dtype=vertices.dtype)
            texel_order = 0

        if (edge_details is None) != (face_edge_ids is None):
            raise ValueError("edge_details and face_edge_ids must be provided together")
        if edge_details is None:
            edge_details = torch.zeros(0, device=vertices.device, dtype=vertices.dtype)
            face_edge_ids = torch.zeros(0, device=vertices.device, dtype=torch.int32)

        # RITS window donors: (window_source [F], donor_indices [D, 3], donor_mode).
        # donor_mode 0 hands the C++ side null pointers, keeping the original code
        # path bit-reproducible exactly like texel_order 0 does for texels.
        if window_donors is None:
            empty = torch.zeros(0, device=vertices.device, dtype=torch.int32)
            window_source, donor_indices, donor_mode = empty, empty, 0
        else:
            window_source, donor_indices, donor_mode = window_donors

        # Per-face window exponents [F]. An empty tensor becomes a null pointer and
        # every face then uses the scheduled scalar `sigma`, exactly as before.
        if sigma_face is None:
            sigma_face = torch.zeros(0, device=vertices.device, dtype=vertices.dtype)

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            vertices,
            triangles_indices,
            vertex_weights,
            sigma,
            sigma_face,
            colors_precomp,
            texels,
            texel_order,
            edge_details,
            face_edge_ids,
            window_source,
            donor_indices,
            donor_mode,
            scaling,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )


        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, radii, was_rendered, geomBuffer, binningBuffer, imgBuffer, scaling, max_blending = _C.rasterize_triangles(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, was_rendered, geomBuffer, binningBuffer, imgBuffer, scaling, max_blending = _C.rasterize_triangles(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.sigma = sigma
        ctx.texel_order = texel_order
        ctx.edge_details_enabled = edge_details.numel() > 0
        ctx.donor_mode = donor_mode
        ctx.per_face_sigma = sigma_face.numel() > 0
        # False reproduces the published backward, whose vertex position gradient
        # carries only the depth term. See cuda_rasterizer/backward.cu.
        ctx.screen_space_gradients = getattr(
            raster_settings, "screen_space_gradients", False)
        ctx.save_for_backward(vertices, triangles_indices, vertex_weights, colors_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, texels, edge_details, face_edge_ids, sigma_face)
        return color, radii, scaling, depth, max_blending, was_rendered

    @staticmethod
    def backward(ctx, grad_out_color, _, __, grad_depth, _____, _______):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        sigma = ctx.sigma
        texel_order = ctx.texel_order
        if ctx.donor_mode != 0:
            raise NotImplementedError(
                "backward through active window donors is not implemented; "
                "RITS-D0 is a forward-only diagnostic (see experiments/rits_d0)")
        vertices, triangles_indices, vertex_weights, colors_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, texels, edge_details, face_edge_ids, sigma_face = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                vertices,
                triangles_indices,
                vertex_weights,
                sigma,
                sigma_face,
                radii, 
                colors_precomp, 
                texels,
                texel_order,
                edge_details,
                face_edge_ids,
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                ctx.screen_space_gradients,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_vertices, grad_vertice_weights, grad_colors_precomp, grad_sh, grad_texels, grad_edge_details, grad_sigma_face = _C.rasterize_triangles_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_vertices, grad_vertice_weights, grad_colors_precomp, grad_sh, grad_texels, grad_edge_details, grad_sigma_face = _C.rasterize_triangles_backward(*args)


        grads = (
            grad_vertices,  
            None,  # triangles_indices
            grad_vertice_weights, # needs to be changed later to vertex_weights
            None, # grad_sigma
            grad_sh,
            grad_colors_precomp,
            None,  # scaling
            None,  # raster_settings
            grad_texels if texel_order > 0 else None,
            grad_edge_details if ctx.edge_details_enabled else None,
            None,  # face_edge_ids
            None,  # window_donors
            grad_sigma_face if ctx.per_face_sigma else None,
        )

        return grads

class TriangleRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    # 0 disables the per-face texel carrier (exact original code path). Defaulted and
    # placed last so existing positional construction of this NamedTuple still works.
    texel_order : int = 0
    # False is the published backward: the vertex position gradient carries only
    # the depth term and dL_dpoints2D is never propagated through the perspective
    # projection. True is the exact derivative, which invalidates the published
    # learning rates and pruning thresholds and so has to be opted into.
    screen_space_gradients : bool = False

class TriangleRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, vertices, triangles_indices, vertex_weights, sigma, scaling,  shs = None, colors_precomp = None, texels = None, edge_details = None, face_edge_ids = None, window_donors = None, sigma_face = None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])


        # Invoke C++/CUDA rasterization routine
        return rasterize_triangles(
            vertices,
            triangles_indices,
            vertex_weights,
            sigma,
            shs,
            colors_precomp,
            scaling,
            raster_settings,
            texels,
            edge_details,
            face_edge_ids,
            window_donors,
            sigma_face,
        )

    @torch.no_grad()
    def forward_with_gorfe_design(
        self,
        vertices,
        triangles_indices,
        vertex_weights,
        sigma,
        scaling,
        gorfe_face_edge_ids,
        gorfe_edge_count,
        output_height,
        output_width,
        output_scaling,
        shs=None,
        colors_precomp=None,
        texels=None,
        edge_details=None,
        face_edge_ids=None,
        window_donors=None,
        sigma_face=None,
    ):
        """Render once, then replay its saved state for GoRFE sparse rows.

        This is an evaluation-only sibling of :meth:`forward`.  The native
        forward is called with exactly the normal argument contract, and only
        after it returns are its opaque buffers passed to ``export_gorfe_rows``.
        The ordinary ``forward`` method and its six-item return remain unchanged.
        """
        if sigma_face is not None:
            raise NotImplementedError(
                "the GoRFE row exporter assumes one window exponent per render; "
                "per-face exponents are not part of its design")
        raster_settings = self.raster_settings
        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise ValueError("provide exactly one of shs or colors_precomp")
        if window_donors is not None:
            raise ValueError("GoRFE design export does not support RITS window donors")
        if texels is not None or (getattr(raster_settings, "texel_order", 0) or 0) != 0:
            raise ValueError("GoRFE design export requires the frozen texel-free parent")
        if edge_details is not None or face_edge_ids is not None:
            raise ValueError("GoRFE design export requires the frozen edge-detail-free parent")
        if output_scaling != 4:
            raise ValueError(
                f"GoRFE-V1 output_scaling must equal 4, got {output_scaling}"
            )
        if raster_settings.image_height != output_height * output_scaling:
            raise ValueError("high-resolution image_height is not exactly 4x output_height")
        if raster_settings.image_width != output_width * output_scaling:
            raise ValueError("high-resolution image_width is not exactly 4x output_width")
        if not isinstance(gorfe_edge_count, int) or gorfe_edge_count < 0:
            raise ValueError("gorfe_edge_count must be a nonnegative integer")

        texel_order = 0
        texels = torch.zeros(0, device=vertices.device, dtype=vertices.dtype)
        edge_details = torch.zeros(0, device=vertices.device, dtype=vertices.dtype)
        face_edge_ids = torch.zeros(0, device=vertices.device, dtype=torch.int32)

        empty = torch.zeros(0, device=vertices.device, dtype=torch.int32)
        window_source, donor_indices, donor_mode = empty, empty, 0
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        args = (
            raster_settings.bg,
            vertices,
            triangles_indices,
            vertex_weights,
            sigma,
            torch.zeros(0, device=vertices.device, dtype=vertices.dtype),
            colors_precomp,
            texels,
            texel_order,
            edge_details,
            face_edge_ids,
            window_source,
            donor_indices,
            donor_mode,
            scaling,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )
        (
            num_rendered,
            color,
            depth,
            radii,
            was_rendered,
            geom_buffer,
            binning_buffer,
            image_buffer,
            output_scaling_tensor,
            max_blending,
        ) = _C.rasterize_triangles(*args)

        pixel_ids, group_ids, features, diagnostics = export_gorfe_rows(
            vertices=vertices,
            triangles_indices=triangles_indices,
            sigma=sigma,
            gorfe_face_edge_ids=gorfe_face_edge_ids,
            gorfe_edge_count=gorfe_edge_count,
            image_height=raster_settings.image_height,
            image_width=raster_settings.image_width,
            output_height=output_height,
            output_width=output_width,
            output_scaling=output_scaling,
            campos=raster_settings.campos,
            geom_buffer=geom_buffer,
            num_rendered=num_rendered,
            binning_buffer=binning_buffer,
            image_buffer=image_buffer,
            debug=raster_settings.debug,
        )
        # `was_rendered` is never passed to the native exporter and cannot
        # influence sparse rows.  It is consulted only after export as an
        # independent saved-forward integrity check required by the V1 gate.
        forward_alpha_accepted = int(
            was_rendered.sum(dtype=torch.int64).detach().cpu().item()
        )
        if diagnostics["count_alpha_accepted_fragments"] != forward_alpha_accepted:
            raise RuntimeError(
                "GoRFE replay alpha-accepted count disagrees with forward: "
                f"{diagnostics['count_alpha_accepted_fragments']} versus "
                f"{forward_alpha_accepted}"
            )
        diagnostics["forward_alpha_accepted_fragments"] = forward_alpha_accepted
        return (
            color,
            radii,
            output_scaling_tensor,
            depth,
            max_blending,
            was_rendered,
            pixel_ids,
            group_ids,
            features,
            diagnostics,
        )


class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)
