#
# Convergence-Gated Regularization (CGR) — per-vertex optimization-trajectory signal.
#
# The fidelity/smoothness dilemma of a single opaque connected mesh is that the
# smoothness a region *needs* is non-monotone in every scalar of the *current
# reconstruction*: a resolved sharp crease and an under-resolved thin protrusion
# both read as high residual and high curvature, yet one must be held rigid and
# the other freed. CGR sidesteps this by reading a functional of the optimization
# *trajectory* instead of a static snapshot of the mesh.
#
# For each vertex we keep two exponential moving averages of the photometric
# position-gradient the optimizer already produces:
#     mu_i : first moment  (a 3-vector)     -- the *coherent* pull
#     nu_i : mean magnitude (a scalar)       -- the *total* pull
# and define the convergence signal
#     O_i = ||mu_i|| / (nu_i + eps)  in [0, 1].
# O_i is LOW where the per-step gradients cancel across views (a multi-view fixed
# point — a resolved crease) and HIGH where the vertex is persistently driven in
# one direction (an under-resolved thin structure). Crucially, O_i lives outside
# the space of static reconstruction scalars, which is why it separates the two
# regimes where residual, curvature and the gradient *magnitude* nu_i all tie.
#
# Nothing in this module participates in autograd — every buffer is a detached
# constant. The uncontaminated photometric gradient is obtained at the rasterizer
# autograd boundary via `photometric_position_gradient`, NOT from `vertices.grad`
# (which, once the full loss has back-propagated, also carries the smoothness
# gradient the gate itself adds — feeding the gate's output back into its input).
#

import numpy as np
import torch


def photometric_position_gradient(loss_image, vertices, retain_graph=True):
    """Return d(loss_image)/d(vertices) — the *pure* photometric position gradient.

    Read at the autograd boundary of the differentiable rasterizer. Because the
    smoothness / normal regularizers are separate terms added to the total loss
    *after* ``loss_image``, they are absent from this graph, so the returned
    gradient is uncontaminated. ``torch.autograd.grad`` does not populate
    ``.grad``, leaving the real optimizer step (``loss.backward()``) untouched —
    no self-referential feedback loop, and no CUDA-kernel change required.
    """
    (grad,) = torch.autograd.grad(
        loss_image, vertices, retain_graph=retain_graph, create_graph=False
    )
    return grad


class CGRTracker:
    """Maintains the per-vertex trajectory EMAs mu_i, nu_i and derives O_i.

    Fixed vertex count for the lifetime of the tracker: topology must be frozen
    while it is active (the diagnostic window disables prune/densify), so buffer
    indices stay in one-to-one correspondence with vertices. Handling remeshing
    (edge collapse/flip/global retriangulation) is a downstream concern; here we
    assert stability rather than silently corrupt the signal.
    """

    def __init__(self, num_vertices, rho=0.9, eps=1e-8, device="cuda"):
        self.rho = float(rho)
        self.eps = float(eps)
        self.device = device
        self.mu = torch.zeros(num_vertices, 3, device=device)
        self.nu = torch.zeros(num_vertices, device=device)
        self.steps = 0

    @property
    def num_vertices(self):
        return self.mu.shape[0]

    @torch.no_grad()
    def update(self, grad_photo):
        """One EMA step from the photometric position gradient ``grad_photo`` (V,3)."""
        if grad_photo.shape[0] != self.num_vertices:
            raise RuntimeError(
                f"CGRTracker vertex count changed ({self.num_vertices} -> "
                f"{grad_photo.shape[0]}): freeze topology while the tracker is active."
            )
        g = grad_photo.detach()
        r = self.rho
        self.mu.mul_(r).add_(g, alpha=1.0 - r)
        self.nu.mul_(r).add_(g.norm(dim=1), alpha=1.0 - r)
        self.steps += 1

    @torch.no_grad()
    def coherence(self):
        """Per-vertex convergence signal O_i = ||mu_i|| / (nu_i + eps) in [0, 1]."""
        return self.mu.norm(dim=1) / (self.nu + self.eps)

    @torch.no_grad()
    def magnitude(self):
        """Per-vertex mean gradient magnitude nu_i — the control signal isolating
        temporal coherence from raw pull strength."""
        return self.nu.clone()

    @torch.no_grad()
    def dump(self, path, vertices, faces):
        """Save the diagnostic signals, reduced to per-face, for the E9 ROC-AUC test.

        ``vertices`` (V,3) and ``faces`` (F,3) are the live geometry. We report
        per-face values (mean over a face's 3 vertices for the trajectory signals)
        alongside a purely-geometric curvature competitor and face centroids, so
        the analysis can transfer ground-truth regime labels by proximity.
        """
        verts = vertices.detach()
        tri = faces.detach().long()
        o_vert = self.coherence()
        nu_vert = self.nu

        o_face = o_vert[tri].mean(dim=1)
        nu_face = nu_vert[tri].mean(dim=1)
        curv_face = per_face_curvature(verts, tri)
        centroid = verts[tri].mean(dim=1)

        np.savez(
            path,
            steps=self.steps,
            rho=self.rho,
            # per-vertex
            vert_coherence=o_vert.cpu().numpy(),
            vert_magnitude=nu_vert.cpu().numpy(),
            vertices=verts.cpu().numpy(),
            faces=tri.cpu().numpy(),
            # per-face signals (the AUC candidates + geometry for label transfer)
            face_coherence=o_face.cpu().numpy(),      # O_i   (ours)
            face_magnitude=nu_face.cpu().numpy(),      # nu_i  (control)
            face_curvature=curv_face.cpu().numpy(),    # curvature (static competitor)
            face_centroid=centroid.cpu().numpy(),
        )
        return path


@torch.no_grad()
def per_face_curvature(vertices, faces):
    """A cheap per-face curvature: the mean dihedral angle to edge-adjacent faces.

    High on sharp creases and on the rims of thin structures; low on flat regions.
    This is the static geometric competitor that tied with the residual signals in
    our earlier stratified experiments — the signal O_i must beat to earn its keep.
    """
    verts = vertices
    tri = faces.long()
    v0, v1, v2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-12)
    num_faces = tri.shape[0]

    # Map undirected edge -> the faces that own it, then accumulate 1 - cos(angle)
    # between the normals of every pair of faces sharing an edge.
    edges = torch.cat([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], dim=0)
    edges, _ = edges.sort(dim=1)
    face_of_edge = torch.arange(num_faces, device=verts.device).repeat(3)

    order = torch.argsort(edges[:, 0] * (verts.shape[0] + 1) + edges[:, 1])
    edges = edges[order]
    face_of_edge = face_of_edge[order]
    same = (edges[1:] == edges[:-1]).all(dim=1)          # adjacent rows share an edge
    pair = same.nonzero(as_tuple=False).squeeze(1)

    curv_sum = torch.zeros(num_faces, device=verts.device)
    curv_cnt = torch.zeros(num_faces, device=verts.device)
    if pair.numel() > 0:
        fa = face_of_edge[pair]
        fb = face_of_edge[pair + 1]
        cos = (normals[fa] * normals[fb]).sum(dim=1).clamp(-1.0, 1.0)
        dihedral = 1.0 - cos                              # 0 flat .. 2 fully folded
        curv_sum.index_add_(0, fa, dihedral)
        curv_sum.index_add_(0, fb, dihedral)
        curv_cnt.index_add_(0, fa, torch.ones_like(dihedral))
        curv_cnt.index_add_(0, fb, torch.ones_like(dihedral))
    return curv_sum / curv_cnt.clamp_min(1.0)
