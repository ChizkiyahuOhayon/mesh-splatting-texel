"""Window-donor construction for the RITS refinement-invariance probes."""

import torch

from csu_split import install_midpoint_probe, midpoint_split_topology, midpoint_values

DONOR_WINDOW = 1
DONOR_OPACITY = 2
DONOR_APPEARANCE = 4


def donor_mode(donor_window, donor_opacity, donor_appearance=False):
    """Encode the mechanism groups as the rasterizer's donor bitmask."""
    return (
        (DONOR_WINDOW if donor_window else 0)
        | (DONOR_OPACITY if donor_opacity else 0)
        | (DONOR_APPEARANCE if donor_appearance else 0)
    )


def install_prolongation_probe(
    model, selected_faces, donor_window, donor_opacity, donor_appearance=False
):
    """Install the inherited midpoint split together with parent window donors.

    ``midpoint_split_topology`` appends the children after the kept faces in
    blocks of four per parent, ordered by the sorted unique parent index; each
    block therefore points at row ``k`` of the parents' original corner
    vertices, which remain valid rows of the vertex buffer after the split.
    """
    selected = torch.unique(selected_faces.long(), sorted=True)
    parents = model._triangle_indices[selected].to(torch.int32).clone()
    probe = install_midpoint_probe(model, selected_faces)

    device = model._triangle_indices.device
    face_count = model._triangle_indices.shape[0]
    kept = probe["base_face_count"] - selected.numel()
    window_source = torch.full((face_count,), -1, dtype=torch.int32, device=device)
    window_source[kept:] = torch.arange(
        selected.numel(), dtype=torch.int32, device=device
    ).repeat_interleave(4)

    mode = donor_mode(donor_window, donor_opacity, donor_appearance)
    probe["window_donors"] = (
        (window_source.contiguous(), parents.to(device).contiguous(), mode)
        if mode
        else None
    )
    probe["child_face_start"] = int(kept)
    return probe


def fd_probe_indices(gradient, count=8):
    """Flattened indices of the largest-magnitude gradient entries.

    Probing the largest entries keeps each per-scalar loss change as far
    above the measurement floor as the gradient allows; uniformly sampled
    scalars are mostly invisible in any single view and carry no signal.
    """
    flat = gradient.detach().abs().flatten()
    return torch.topk(flat, min(count, flat.numel())).indices


def fd_rung_check(analytic, coarse, fine, tolerance=0.05, growth=1.25):
    """Two-rung central-difference check against an analytic gradient.

    The fine-step estimate must agree within `tolerance` and its error must
    not exceed `growth` times the coarse-step error: a correct gradient is
    the small-step limit of the difference quotient, so halving the step must
    not move the estimate away from it beyond noise.
    """
    scale = max(abs(analytic), abs(fine), 1e-12)
    relative = abs(fine - analytic) / scale
    converging = abs(fine - analytic) <= growth * abs(coarse - analytic)
    return {
        "relative": relative,
        "converging": converging,
        "pass": relative <= tolerance and converging,
    }


# Restore-path fine-tuning learning rates (scene/triangle_model.py
# load_parameters); vertex opacity stays frozen by convention.
FINETUNE_LRS = {"f_dc": 0.0016, "f_rest": 0.0016 / 20.0, "vertices": 0.0001, "vertex_weight": 0.0}


def install_trainable_split(model, selected_faces):
    """Install the midpoint split with every parameter trainable.

    Unlike ``install_prolongation_probe``, which exposes only the new midpoint
    values as autograd leaves for read-only probes, this concatenates original
    and midpoint values into whole leaf tensors and rebuilds a fresh Adam with
    the restore-path learning rates, so fine-tuning arms share one optimizer
    construction. Returns the RITS-T0 state: full donors (mode 7), the first
    child row, and the original tensor sizes for gradient-slice checks.
    """
    if getattr(model, "texel_order", 0) != 0:
        raise ValueError("RITS-T0 requires an SH-only model")

    selected = torch.unique(selected_faces.long(), sorted=True)
    parents = model._triangle_indices[selected].to(torch.int32).clone()
    base_vertex_count = model.vertices.shape[0]
    base_face_count = model._triangle_indices.shape[0]
    topology = midpoint_split_topology(
        model._triangle_indices, selected_faces, base_vertex_count
    )
    edges = topology["unique_edges"]

    new_vertices = midpoint_values(model.vertices.detach(), edges)
    new_dc = midpoint_values(model._features_dc.detach(), edges)
    new_rest = midpoint_values(model._features_rest.detach(), edges)
    opacity = model.opacity_activation(model.vertex_weight.detach())
    new_weight = model.inverse_opacity_activation(
        midpoint_values(opacity, edges).clamp(
            model.opacity_floor + model.eps, 1.0 - model.eps
        )
    )

    model.optimizer = None
    model.vertices = torch.cat(
        (model.vertices.detach(), new_vertices), dim=0
    ).requires_grad_(True)
    model.vertex_weight = torch.cat(
        (model.vertex_weight.detach(), new_weight), dim=0
    ).requires_grad_(True)
    model._features_dc = torch.cat(
        (model._features_dc.detach(), new_dc), dim=0
    ).requires_grad_(True)
    model._features_rest = torch.cat(
        (model._features_rest.detach(), new_rest), dim=0
    ).requires_grad_(True)
    model._triangle_indices = topology["triangle_indices"]

    device = model.vertices.device
    face_count = model._triangle_indices.shape[0]
    model.image_size = torch.zeros(face_count, dtype=torch.float, device=device)
    model.importance_score = torch.zeros(face_count, dtype=torch.float, device=device)
    model.pixel_count = torch.zeros(face_count, dtype=torch.int, device=device)
    model.validate_face_state()

    kept = base_face_count - selected.numel()
    window_source = torch.full((face_count,), -1, dtype=torch.int32, device=device)
    window_source[kept:] = torch.arange(
        selected.numel(), dtype=torch.int32, device=device
    ).repeat_interleave(4)

    tensors = {
        "f_dc": model._features_dc,
        "f_rest": model._features_rest,
        "vertices": model.vertices,
        "vertex_weight": model.vertex_weight,
    }
    model.optimizer = torch.optim.Adam(
        [
            {"params": [tensor], "lr": FINETUNE_LRS[name], "name": name}
            for name, tensor in tensors.items()
        ],
        lr=0.0,
        eps=1e-15,
    )

    return {
        "window_donors": (
            window_source.contiguous(),
            parents.to(device).contiguous(),
            donor_mode(True, True, True),
        ),
        "child_face_start": int(kept),
        "base_vertex_count": int(base_vertex_count),
        "base_face_count": int(base_face_count),
        "split_vertex_count": int(model.vertices.shape[0]),
        "split_face_count": int(face_count),
        "unique_edge_count": int(edges.shape[0]),
        "selected_faces": selected,
    }
