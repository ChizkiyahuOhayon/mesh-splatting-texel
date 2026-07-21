# Server setup and experiment protocol

Fork of MeshSplatting (CVPR 2026) adding a **per-face texel appearance carrier**.

All three submodules (`diff-triangle-mesh-rasterization`, `effrdel`, `simple-knn`) are
**vendored directly into this repo** — no `git submodule` dance, and the tree does not
depend on `gitlab.inria.fr` staying up. `git clone` gives a complete, buildable tree.

---

## 0. Clone (private repo)

A private repo clones fine; it just needs auth. On the A40, once:

```bash
gh auth login          # choose SSH or HTTPS+token
# or, without gh:  git clone https://<TOKEN>@github.com/<user>/<repo>.git
```

Then:

```bash
cd /home/smbu/dy
git clone git@github.com:<user>/<repo>.git mesh-splatting-texel
cd mesh-splatting-texel
```

## 1. Rebuild the CUDA extension — REQUIRED

The rasterizer's `.cu` / `.h` sources changed, so the compiled `.so` **must** be rebuilt.
Skipping this silently runs the OLD binary and every result is meaningless.

```bash
micromamba activate mesh_splatting   # or: conda activate mesh_splatting
cd /home/smbu/dy/mesh-splatting-texel

# only the rasterizer changed; effrdel / simple-knn can be reused if already installed
pip uninstall -y diff_triangle_rasterization
pip install submodules/diff-triangle-mesh-rasterization --no-build-isolation
```

If `effrdel` / `simple-knn` are not yet installed in this env:

```bash
pip install pybind11
pip install submodules/effrdel   --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
```

Confirm the freshly built extension is the one being imported:

```bash
python -c "import diff_triangle_rasterization as d; print(d.__file__)"
```

## 2. Verify before training — REQUIRED

The CUDA changes were written without access to an NVIDIA GPU. **Run this first.**
It takes seconds; a garden training run takes ~1.5 h.

```bash
python verify_texel.py -s /home/smbu/dy/mesh-splatting/data/mipnerf360/garden --order 2
```

Checks:

| | what | why it matters |
|---|---|---|
| T1 | `texel_order=0` renders normally | the disabled path must be the original code path |
| T2 | order 2 with zero texels renders **bit-identically** to order 0 | the carrier is introduced zero-initialised; if this is not exactly 0 the indexing or the additive term is wrong, and no later comparison is attributable |
| T3 | analytic `dL/dtexel` matches finite differences (<5%) | the backward pass is correct |

**If any check fails, stop and send me the output.** Do not start training.

## 3. Experiments

### E3-A — regularization sweep, baseline carrier (no code change needed)

Produces the per-vertex-SH arm of the identification experiment. Can run immediately.

```bash
bash bash_scripts/exp_lambda_sweep_sh.sh \
     /home/smbu/dy/mesh-splatting/data/mipnerf360/garden \
     output/lambda_sweep
```

**Control point:** the `x1` run is the stock configuration and should reproduce the
original garden result (**PSNR ≈ 24.71**, per `record1.md`). If it does not, stop —
something is wrong with the environment, not with the method.

### E3-B — texel carrier vs baseline, end-to-end

```bash
# baseline (identical to upstream; texel_order defaults to 0)
python train.py -s <garden> -m output/texel/baseline --eval

# texel carrier, order 2 (4 texels/face), introduced after the Delaunay retriangulation
python train.py -s <garden> -m output/texel/order2 --eval --texel_order 2

# order 3 (9 texels/face)
python train.py -s <garden> -m output/texel/order3 --eval --texel_order 3
```

Send me, for each run: the final test metrics, the `[texel] allocated ...` line, the
final triangle/vertex counts, and the wall-clock time.

---

## What changed vs upstream

Appearance in MeshSplatting is carried by **vertices** (SH interpolated barycentrically),
which welds the spatial frequency of appearance to tessellation density. This fork adds an
**additive per-face texel residual**:

```
color = barycentric(vertex SH colour) + texels[face_id, slot]
```

- Each triangle is subdivided barycentrically into `order^2` texels (Ptex-style, no UV
  unwrap). Lookup is nearest, i.e. `GL_NEAREST`-equivalent.
- The carrier is allocated **immediately after `run_restricted_delaunay()`**, because that
  operation rebuilds the whole face set (per-face values have no correspondence across
  it), and because densification stops before it, so the face count is stable afterwards.
- It is **zero-initialised**, so at the moment of introduction the model is numerically
  identical to the baseline.
- It lives in its **own optimizer**. The main optimizer's prune/densify helpers apply a
  single per-*vertex* mask to every param group, which would silently corrupt a
  per-*face* tensor. Triangle pruning syncs the texels explicitly.
- `--texel_order 0` (the default) passes a null pointer to CUDA and takes exactly the
  original code path, so this repo still reproduces upstream bit-for-bit.

Nearest (rather than linear) lookup is deliberate for this first version: a piecewise
constant lookup has zero derivative w.r.t. the barycentric coordinates almost everywhere,
so the backward pass leaves all existing barycentric/geometry gradient paths untouched.

Files changed: `cuda_rasterizer/{texel.h (new),forward.cu,forward.h,backward.cu,backward.h,
rasterizer.h,rasterizer_impl.cu}`, `rasterize_points.{cu,h}`,
`diff_triangle_rasterization/__init__.py`, `scene/triangle_model.py`,
`triangle_renderer/__init__.py`, `train.py`, `arguments/__init__.py`.
