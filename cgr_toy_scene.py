"""CGR (Convergence-Gated Regularization) — E9 controlled toy scene generator.

Builds a *known* ground-truth mesh that deliberately contains the two regimes the
CGR separation-diagnostic must tell apart:

  * RESOLVED-CREASE geometry  — the sharp 90-degree dihedral edges of a central
    cube. Multi-view photometric gradients at a well-fit crease *cancel* (the
    crease sits at a multi-view fixed point), so the trajectory-coherence signal
    O_i should read LOW there.
  * UNDER-RESOLVED-THIN geometry — thin protruding fins (a few edge-lengths
    thick). A fin that the reconstruction has not yet grown is *persistently
    driven* in one direction, so O_i should read HIGH there.

The generator emits a MeshSplatting-ingestible **Blender / NeRF-synthetic** scene
(``transforms_train.json`` + ``transforms_test.json`` + RGBA PNGs), plus the GT
mesh and a per-GT-face regime label array. E9 (``cgr_auc.py``, separate module)
transfers these labels onto the reconstructed faces by proximity and computes the
ROC-AUC of each candidate signal.

Everything here is pure numpy + trimesh + PIL — no GL, no Embree, no extra deps —
so it runs on the Mac M4 for $0 and is deterministic given ``--seed``. Ground-truth
views are produced by a self-contained, chunked, vectorized Moeller-Trumbore
raycaster with a Lambertian shading model and a high-frequency procedural albedo
(the texture is what makes vertex position photometrically observable — a flat
untextured surface would yield ~zero lateral position gradient and no signal).

Face regime labels (per GT face): 0 = flat, 1 = crease, 2 = thin.

Usage
-----
    python maclab/cgr_toy_scene.py --out data/cgr_toy --n-train 100 --n-test 20 \
        --res 400 --seed 0
    # quick local smoke test:
    python maclab/cgr_toy_scene.py --out /tmp/cgr_toy_smoke --n-train 6 --n-test 2 \
        --res 128 --seed 0 --preview
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import trimesh
from PIL import Image

# Regime label codes (per GT face).
LABEL_FLAT = 0
LABEL_CREASE = 1
LABEL_THIN = 2
LABEL_NAMES = {LABEL_FLAT: "flat", LABEL_CREASE: "crease", LABEL_THIN: "thin"}


# --------------------------------------------------------------------------- #
# Mesh construction
# --------------------------------------------------------------------------- #
def _grid_plane(origin, u_vec, v_vec, nu, nv):
    """Tessellate a rectangle spanned by ``u_vec`` x ``v_vec`` from ``origin``.

    Returns (verts [(nu+1)(nv+1), 3], faces [2*nu*nv, 3]) with outward winding
    following the right-hand rule of (u_vec, v_vec).
    """
    origin = np.asarray(origin, dtype=np.float64)
    u_vec = np.asarray(u_vec, dtype=np.float64)
    v_vec = np.asarray(v_vec, dtype=np.float64)
    su = np.linspace(0.0, 1.0, nu + 1)
    sv = np.linspace(0.0, 1.0, nv + 1)
    gu, gv = np.meshgrid(su, sv, indexing="ij")  # (nu+1, nv+1)
    verts = (
        origin[None, None, :]
        + gu[..., None] * u_vec[None, None, :]
        + gv[..., None] * v_vec[None, None, :]
    ).reshape(-1, 3)

    def vid(i, j):
        return i * (nv + 1) + j

    faces = []
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, np.asarray(faces, dtype=np.int64)


def _box_grid(center, half, res):
    """A cube of half-extent ``half`` centered at ``center``, each of the 6 faces
    tessellated ``res`` x ``res``. Returns (verts, faces) with outward normals.
    """
    cx, cy, cz = center
    hx, hy, hz = half
    planes = [
        # (origin, u_vec, v_vec) — winding chosen for outward normals
        ([cx - hx, cy - hy, cz + hz], [2 * hx, 0, 0], [0, 2 * hy, 0]),   # +Z
        ([cx - hx, cy + hy, cz - hz], [2 * hx, 0, 0], [0, -2 * hy, 0]),  # -Z
        ([cx + hx, cy - hy, cz - hz], [0, 2 * hy, 0], [0, 0, 2 * hz]),   # +X
        ([cx - hx, cy + hy, cz - hz], [0, -2 * hy, 0], [0, 0, 2 * hz]),  # -X
        ([cx - hx, cy + hy, cz - hz], [2 * hx, 0, 0], [0, 0, 2 * hz]),   # +Y
        ([cx - hx, cy - hy, cz + hz], [2 * hx, 0, 0], [0, 0, -2 * hz]),  # -Y
    ]
    all_v, all_f, off = [], [], 0
    for origin, u, v in planes:
        vs, fs = _grid_plane(origin, u, v, res, res)
        all_v.append(vs)
        all_f.append(fs + off)
        off += len(vs)
    return np.concatenate(all_v, 0), np.concatenate(all_f, 0)


@dataclass
class ToyMesh:
    mesh: trimesh.Trimesh
    face_label: np.ndarray  # (F,) in {0,1,2}


def build_toy_mesh(seed=0, cube_res=14, fin_res=10, crease_band=0.08):
    """Central cube (creases) + four thin fins (thin structures).

    ``crease_band`` (in world units) is how close a cube face's centroid must be
    to a cube edge to be labelled a crease; interior cube faces are ``flat``.
    """
    rng = np.random.default_rng(seed)
    cube_half = np.array([0.5, 0.5, 0.5])
    v_list, f_list, lab_list, off = [], [], [], 0

    # --- central cube -----------------------------------------------------
    cv, cf = _box_grid([0, 0, 0], cube_half, cube_res)
    v_list.append(cv)
    f_list.append(cf + off)
    off += len(cv)
    # label cube faces: crease if centroid is within crease_band of any cube edge.
    tri = cv[cf]  # (F,3,3)
    cent = tri.mean(axis=1)  # (F,3)
    # distance to nearest cube edge = for a point on a box surface, the two
    # in-face coordinates' distance to +/- half. Compute per-axis gap to the box
    # boundary and take, on each face, the min over the two tangential axes.
    gap = cube_half[None, :] - np.abs(cent)  # (F,3) small near a boundary plane
    # the axis with the largest |cent| ~ the face normal axis (gap~0 there means
    # the face itself); the crease proximity is the min gap over the OTHER axes.
    normal_axis = np.argmax(np.abs(cent), axis=1)  # (F,)
    cube_lab = np.full(len(cf), LABEL_FLAT, dtype=np.int64)
    for fi in range(len(cf)):
        tang = [a for a in range(3) if a != normal_axis[fi]]
        edge_dist = gap[fi, tang].min()
        if edge_dist <= crease_band:
            cube_lab[fi] = LABEL_CREASE
    lab_list.append(cube_lab)

    # --- thin cylindrical spokes (the under-resolved-thin regime) ---------
    # 2D-thin tubes are far harder for an opaque connected mesh than a flat plate:
    # the mesh must wrap around a small-radius cylinder, so sub-resolution spokes
    # stay under-resolved (persistently driven) while the cube's creases resolve.
    # Varying radius spans clearly-resolvable to near the pixel limit.
    spoke_specs = [
        # (base_on_cube, out_dir, length, radius)
        ([0.5, 0.15, 0.1], [1, 0, 0], 0.7, 0.035),
        ([0.5, -0.2, -0.15], [1, 0.15, 0], 0.7, 0.020),
        ([-0.5, 0.1, 0.15], [-1, 0, 0.1], 0.7, 0.028),
        ([-0.5, -0.15, -0.1], [-1, 0, 0], 0.7, 0.014),
        ([0.1, 0.5, 0.1], [0, 1, 0], 0.7, 0.030),
        ([-0.1, 0.5, -0.15], [0.1, 1, 0], 0.7, 0.017),
        ([0.15, -0.5, 0.1], [0, -1, 0], 0.7, 0.024),
        ([-0.1, -0.5, -0.1], [0, -1, 0.1], 0.7, 0.012),
    ]
    for (base, out_dir, length, radius) in spoke_specs:
        sv, sf = _spoke(base, out_dir, length, radius, sections=16)
        v_list.append(sv)
        f_list.append(sf + off)
        off += len(sv)
        lab_list.append(np.full(len(sf), LABEL_THIN, dtype=np.int64))

    verts = np.concatenate(v_list, 0)
    faces = np.concatenate(f_list, 0)
    face_label = np.concatenate(lab_list, 0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Weld coincident vertices so the mesh is a proper manifold with shared edges
    # across the cube faces (its sharp creases). merge_vertices preserves face order
    # and count, so the per-face labels stay aligned.
    mesh.merge_vertices()
    assert len(mesh.faces) == len(face_label), "weld must preserve face order/count"
    return ToyMesh(mesh=mesh, face_label=face_label)


def _spoke(base, direction, length, radius, sections=16):
    """A thin cylindrical spoke of ``radius`` from ``base`` along ``direction``.

    Returns (verts, faces). The base is sunk slightly into the cube so the spoke
    joins the surface rather than floating."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    base = np.asarray(base, dtype=np.float64) - direction * 0.05  # sink into the cube
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], direction))
    cyl.apply_translation(base + direction * (length / 2.0))
    return np.asarray(cyl.vertices, dtype=np.float64), np.asarray(cyl.faces, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Procedural albedo (spatial texture => position is photometrically observable)
# --------------------------------------------------------------------------- #
def face_albedo(mesh: trimesh.Trimesh, seed=0):
    """Per-face RGB albedo from a high-frequency 3D pattern on face centroids."""
    cent = mesh.triangles.mean(axis=1)  # (F,3)
    freq = np.array([7.0, 9.0, 11.0])
    phase = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=3)
    base = 0.5 + 0.35 * np.stack(
        [np.sin(freq[k] * cent[:, k % 3] + freq[(k + 1) % 3] * cent[:, (k + 1) % 3] + phase[k])
         for k in range(3)], axis=1
    )
    return np.clip(base, 0.05, 1.0)


# --------------------------------------------------------------------------- #
# Cameras (Blender / NeRF-synthetic convention: c2w, OpenGL axes)
# --------------------------------------------------------------------------- #
def sample_cameras(n, radius=3.0, seed=0, elev_range=(-40.0, 60.0)):
    """Sample ``n`` camera-to-world matrices on a sphere looking at the origin.

    Returns list of 4x4 c2w in OpenGL/Blender convention (camera looks down -Z,
    +Y up) — exactly what ``readCamerasFromTransforms`` expects.
    """
    rng = np.random.default_rng(seed + 12345)
    c2ws = []
    for _ in range(n):
        az = rng.uniform(0, 2 * np.pi)
        el = np.deg2rad(rng.uniform(*elev_range))
        cam_pos = radius * np.array(
            [np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)]
        )
        forward = -cam_pos / np.linalg.norm(cam_pos)   # look at origin
        up_world = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up_world)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        # OpenGL camera basis: x=right, y=up, z=-forward (back)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos
        c2ws.append(c2w)
    return c2ws


# --------------------------------------------------------------------------- #
# Self-contained chunked vectorized Moeller-Trumbore raycaster + shading
# --------------------------------------------------------------------------- #
def _render_view(verts, faces, albedo, c2w, fovx, H, W, lights, ambient=0.25,
                 chunk=16384):
    """Return (rgb [H,W,3] float in [0,1], alpha [H,W] float) for one camera.

    Vectorized Moeller-Trumbore. Two structural optimizations keep it fast enough
    to run on a laptop CPU: (1) the ray *origin* is shared across all pixels of a
    pinhole camera, so ``tvec = origin - v0`` and ``qvec = cross(tvec, e1)`` are
    ray-INDEPENDENT (computed once per face, not per ray); the only (r,F,3)
    temporary is ``pvec``. (2) A cheap ray/bounding-sphere test culls background
    rays before the O(r*F) face intersection. float32 throughout.
    """
    verts = verts.astype(np.float32)
    tri = verts[faces]  # (F,3,3)
    v0 = tri[:, 0, :]
    e1 = tri[:, 1, :] - v0
    e2 = tri[:, 2, :] - v0
    face_n = np.cross(e1, e2)
    face_n /= (np.linalg.norm(face_n, axis=1, keepdims=True) + 1e-12)

    # pixel ray directions in camera space (OpenGL: looks down -Z)
    focal = 0.5 * W / np.tan(0.5 * fovx)
    ys, xs = np.mgrid[0:H, 0:W]
    px = (xs + 0.5 - W / 2.0) / focal
    py = -(ys + 0.5 - H / 2.0) / focal
    dirs_cam = np.stack([px, py, -np.ones_like(px)], axis=-1).reshape(-1, 3)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    R = c2w[:3, :3].astype(np.float32)
    origin = c2w[:3, 3].astype(np.float32)
    dirs = (dirs_cam @ R.T).astype(np.float32)  # (R,3) world
    R_rays = dirs.shape[0]

    # bounding-sphere cull: only rays whose closest approach to the scene center
    # is within the bounding radius can hit the object.
    center = verts.reshape(-1, 3).mean(0)
    bsph = np.linalg.norm(verts.reshape(-1, 3) - center, axis=1).max() * 1.02
    oc = center[None, :] - origin[None, :]                     # (1,3)
    tca = np.einsum("rk,k->r", dirs, oc[0])                     # proj of oc on ray
    d2 = np.einsum("rk,rk->r", dirs, dirs) * 0.0 + (
        np.linalg.norm(oc) ** 2 - tca ** 2)                    # perp dist^2
    candidate = d2 <= bsph ** 2

    # ray-independent per-face quantities
    tvec = (origin[None, :] - v0)                              # (F,3)
    qvec = np.cross(tvec, e1)                                  # (F,3)
    e2_dot_q = np.einsum("fk,fk->f", e2, qvec)                 # (F,)

    best_f = np.full(R_rays, -1, dtype=np.int64)
    eps = np.float32(1e-8)
    cand_idx = np.where(candidate)[0]
    for s in range(0, cand_idx.size, chunk):
        ridx = cand_idx[s:s + chunk]
        d = dirs[ridx]                                         # (r,3)
        pvec = np.cross(d[:, None, :], e2[None, :, :])         # (r,F,3)
        det = np.einsum("fk,rfk->rf", e1, pvec)                # (r,F)
        inv_det = np.where(np.abs(det) > eps, 1.0 / det, 0.0)
        u = np.einsum("fk,rfk->rf", tvec, pvec) * inv_det      # (r,F)
        v = np.einsum("rk,fk->rf", d, qvec) * inv_det          # (r,F)
        t = e2_dot_q[None, :] * inv_det                        # (r,F)
        valid = (np.abs(det) > eps) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-5)
        t_masked = np.where(valid, t, np.inf)
        fmin = np.argmin(t_masked, axis=1)
        tmin = t_masked[np.arange(ridx.size), fmin]
        hit = np.isfinite(tmin)
        best_f[ridx] = np.where(hit, fmin, -1)

    alpha = (best_f >= 0).astype(np.float64)
    rgb = np.zeros((R_rays, 3))
    hit_idx = np.where(best_f >= 0)[0]
    if hit_idx.size:
        fi = best_f[hit_idx]
        n = face_n[fi]
        # two-sided normal (face the camera)
        vdir = -dirs[hit_idx]
        flip = (np.einsum("rk,rk->r", n, vdir) < 0)[:, None]
        n = np.where(flip, -n, n)
        shade = np.full(hit_idx.shape[0], ambient)
        for L in lights:
            ld = L / np.linalg.norm(L)
            shade = shade + np.clip(np.einsum("rk,k->r", n, ld), 0, None) * (0.75 / len(lights))
        rgb[hit_idx] = albedo[fi] * shade[:, None]
    rgb = np.clip(rgb, 0, 1).reshape(H, W, 3)
    alpha = alpha.reshape(H, W)
    return rgb, alpha


# --------------------------------------------------------------------------- #
# Scene writer
# --------------------------------------------------------------------------- #
def _write_split(outdir, split, c2ws, verts, faces, albedo, fovx, H, W, lights):
    img_dir = os.path.join(outdir, split)
    os.makedirs(img_dir, exist_ok=True)
    frames = []
    for i, c2w in enumerate(c2ws):
        rgb, alpha = _render_view(verts, faces, albedo, c2w, fovx, H, W, lights)
        rgba = np.dstack([rgb, alpha])
        Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(
            os.path.join(img_dir, f"r_{i}.png")
        )
        frames.append({
            "file_path": f"./{split}/r_{i}",
            "transform_matrix": c2w.tolist(),
        })
    with open(os.path.join(outdir, f"transforms_{split}.json"), "w") as f:
        json.dump({"camera_angle_x": float(fovx), "frames": frames}, f, indent=2)
    return len(frames)


def generate(outdir, n_train=100, n_test=20, res=400, fovx_deg=45.0, seed=0,
             radius=3.0, cube_res=14, fin_res=10, preview=False):
    os.makedirs(outdir, exist_ok=True)
    toy = build_toy_mesh(seed=seed, cube_res=cube_res, fin_res=fin_res)
    verts = np.asarray(toy.mesh.vertices, dtype=np.float64)
    faces = np.asarray(toy.mesh.faces, dtype=np.int64)
    albedo = face_albedo(toy.mesh, seed=seed)
    fovx = np.deg2rad(fovx_deg)
    lights = [np.array([1.0, 1.5, 0.8]), np.array([-1.0, 0.5, -1.2])]

    train_c2ws = sample_cameras(n_train, radius=radius, seed=seed)
    test_c2ws = sample_cameras(n_test, radius=radius, seed=seed + 777)

    n_tr = _write_split(outdir, "train", train_c2ws, verts, faces, albedo, fovx, res, res, lights)
    n_te = _write_split(outdir, "test", test_c2ws, verts, faces, albedo, fovx, res, res, lights)

    # GT mesh + per-face labels + centroids (label transfer substrate for E9)
    toy.mesh.export(os.path.join(outdir, "gt_mesh.ply"))
    np.savez(
        os.path.join(outdir, "gt_labels.npz"),
        face_label=toy.face_label,
        face_centroid=toy.mesh.triangles.mean(axis=1),
        face_normal=np.asarray(toy.mesh.face_normals),
        vertices=verts,
        faces=faces,
    )
    counts = {LABEL_NAMES[k]: int((toy.face_label == k).sum()) for k in LABEL_NAMES}
    meta = {
        "n_train": n_tr, "n_test": n_te, "res": res, "fovx_deg": fovx_deg,
        "seed": seed, "n_faces": int(len(faces)), "n_verts": int(len(verts)),
        "label_counts": counts, "radius": radius,
    }
    with open(os.path.join(outdir, "cgr_toy_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if preview:
        rgb, alpha = _render_view(verts, faces, albedo, train_c2ws[0], fovx, res, res, lights)
        Image.fromarray((np.dstack([rgb, alpha]) * 255).astype(np.uint8), "RGBA").save(
            os.path.join(outdir, "preview_view0.png")
        )
    print(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description="CGR E9 controlled toy scene generator")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--fovx-deg", type=float, default=45.0)
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--cube-res", type=int, default=14)
    ap.add_argument("--fin-res", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    generate(
        args.out, n_train=args.n_train, n_test=args.n_test, res=args.res,
        fovx_deg=args.fovx_deg, seed=args.seed, radius=args.radius,
        cube_res=args.cube_res, fin_res=args.fin_res, preview=args.preview,
    )


if __name__ == "__main__":
    main()
