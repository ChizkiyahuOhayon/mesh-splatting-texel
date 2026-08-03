"""Window-donor construction for the RITS refinement-invariance probes."""

import torch

from csu_split import install_midpoint_probe

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
