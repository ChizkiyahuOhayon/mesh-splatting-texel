"""Topology and parameter helpers for the CSU-F0 midpoint-split probe."""

import torch


def midpoint_split_topology(triangle_indices, selected_faces, vertex_count):
    """Replace selected parents by four children with shared edge midpoints."""
    if triangle_indices.ndim != 2 or triangle_indices.shape[1] != 3:
        raise ValueError("triangle_indices must have shape [F, 3]")
    if selected_faces.ndim != 1 or selected_faces.numel() == 0:
        raise ValueError("selected_faces must be a non-empty 1-D tensor")

    selected = torch.unique(selected_faces.long(), sorted=True)
    if selected[0] < 0 or selected[-1] >= triangle_indices.shape[0]:
        raise IndexError("selected face index is out of range")

    parents = triangle_indices[selected].long()
    edges = torch.stack(
        (parents[:, [0, 1]], parents[:, [0, 2]], parents[:, [1, 2]]), dim=1
    )
    edges = torch.sort(edges.reshape(-1, 2), dim=1).values
    unique_edges, inverse = torch.unique(edges, dim=0, return_inverse=True)
    midpoints = inverse.reshape(-1, 3) + int(vertex_count)

    a, b, c = parents.unbind(dim=1)
    midpoint_ab, midpoint_ac, midpoint_bc = midpoints.unbind(dim=1)
    children = torch.stack(
        (
            torch.stack((a, midpoint_ab, midpoint_ac), dim=1),
            torch.stack((b, midpoint_ab, midpoint_bc), dim=1),
            torch.stack((c, midpoint_ac, midpoint_bc), dim=1),
            torch.stack((midpoint_ab, midpoint_bc, midpoint_ac), dim=1),
        ),
        dim=1,
    ).reshape(-1, 3)

    keep = torch.ones(
        triangle_indices.shape[0], dtype=torch.bool, device=triangle_indices.device
    )
    keep[selected] = False
    updated = torch.cat((triangle_indices[keep].long(), children), dim=0)
    return {
        "selected_faces": selected,
        "unique_edges": unique_edges,
        "triangle_indices": updated.to(triangle_indices.dtype).contiguous(),
    }


def midpoint_values(values, unique_edges):
    """Interpolate one value at each unique edge midpoint."""
    if values.ndim < 1:
        raise ValueError("values must have a vertex dimension")
    if unique_edges.ndim != 2 or unique_edges.shape[1] != 2:
        raise ValueError("unique_edges must have shape [E, 2]")
    return 0.5 * (values[unique_edges[:, 0]] + values[unique_edges[:, 1]])


def install_midpoint_probe(model, selected_faces):
    """Install an inherited split while exposing only its new values as leaves."""
    if getattr(model, "texel_order", 0) != 0:
        raise ValueError("CSU-F0 requires an SH-only model")

    base_vertices = model.vertices
    base_weight = model.vertex_weight
    base_dc = model._features_dc
    base_rest = model._features_rest
    base_vertex_count = base_vertices.shape[0]
    base_face_count = model._triangle_indices.shape[0]
    topology = midpoint_split_topology(
        model._triangle_indices, selected_faces, base_vertex_count
    )
    edges = topology["unique_edges"]

    new_vertices = midpoint_values(base_vertices.detach(), edges).clone().requires_grad_(True)
    new_dc = midpoint_values(base_dc.detach(), edges).clone().requires_grad_(True)
    new_rest = midpoint_values(base_rest.detach(), edges).clone().requires_grad_(True)
    opacity = model.opacity_activation(base_weight.detach())
    new_opacity = midpoint_values(opacity, edges).clamp(
        model.opacity_floor + model.eps, 1.0 - model.eps
    )
    new_weight = model.inverse_opacity_activation(new_opacity).clone().requires_grad_(True)

    model.optimizer = None
    model.vertices = torch.cat((base_vertices.detach(), new_vertices), dim=0)
    model.vertex_weight = torch.cat((base_weight.detach(), new_weight), dim=0)
    model._features_dc = torch.cat((base_dc.detach(), new_dc), dim=0)
    model._features_rest = torch.cat((base_rest.detach(), new_rest), dim=0)
    model._triangle_indices = topology["triangle_indices"]

    prefix_unchanged = all(
        (
            torch.equal(model.vertices[:base_vertex_count], base_vertices.detach()),
            torch.equal(model.vertex_weight[:base_vertex_count], base_weight.detach()),
            torch.equal(model._features_dc[:base_vertex_count], base_dc.detach()),
            torch.equal(model._features_rest[:base_vertex_count], base_rest.detach()),
        )
    )
    face_count = model._triangle_indices.shape[0]
    device = model.vertices.device
    model.image_size = torch.zeros(face_count, dtype=torch.float, device=device)
    model.importance_score = torch.zeros(face_count, dtype=torch.float, device=device)
    model.pixel_count = torch.zeros(face_count, dtype=torch.int, device=device)
    model.validate_face_state()

    expected_face_count = base_face_count + 3 * topology["selected_faces"].numel()
    topology_valid = (
        face_count == expected_face_count
        and model.vertices.shape[0] == base_vertex_count + edges.shape[0]
    )
    return {
        "parameters": {
            "vertices": new_vertices,
            "vertex_weight": new_weight,
            "features_dc": new_dc,
            "features_rest": new_rest,
        },
        "selected_faces": topology["selected_faces"],
        "unique_edge_count": int(edges.shape[0]),
        "base_vertex_count": int(base_vertex_count),
        "base_face_count": int(base_face_count),
        "split_vertex_count": int(model.vertices.shape[0]),
        "split_face_count": int(face_count),
        "prefix_unchanged": prefix_unchanged,
        "topology_valid": topology_valid,
    }
