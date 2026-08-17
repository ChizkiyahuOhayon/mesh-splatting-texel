# What could actually beat MeshSplatting

Written 2026-08-17, after the campaign recorded in `sota/PLAN.md` failed. Every
number quoted here was measured on this codebase at revision `c654219`; the raw
logs are in `sota/results/`.

The point of this document is that "we could not beat it" and "it cannot be
beaten" are different claims, and only the first one is supported. **24.78 dB is
low in absolute terms** — its own soft ancestor, Triangle Splatting, is at 27.16
and 3DGS at 27.21. There is 2.4 dB of headroom above it. What is hard about
MeshSplatting is not the height of the number, it is that

1. the opaque-connected-mesh constraint costs that 2.4 dB and the cost is
   structural, not a scheduling artefact, and
2. the published configuration sits at a local optimum where single-component
   perturbations are neutral to negative, and
3. it is simultaneously near its class's quality frontier *and* its compactness
   frontier, so adding capacity costs storage and removing primitives costs
   quality.

Beating it therefore requires enlarging the model class or changing the
objective. Perturbing a component does not work and has now been tried nineteen
times.

---

## Four rules that cost a month to learn

**1. Never make a decision on Garden.** Garden's baseline carries 6,659,278
triangles over 1.09 Mpx — **6.1 triangles per pixel**. Room carries 5,596,666
over 1.62 Mpx — **3.5 per pixel**. Garden is grossly over-tessellated and has
slack to give back, so anything that trades primitives for something else looks
good there and does nothing elsewhere. Three unrelated interventions were
`+0.221`, `+0.618`, `+0.451` capacity-matched on Garden and `−0.005`, `+0.012`,
`−0.049` on Room. July's texel experiment had the same shape a month earlier:
Garden `+0.3`, nine-scene mean `−0.103`. **Minimum viable evidence is three
scenes spanning both protocols, e.g. Garden, Room, Bicycle.**

**2. Always control for model size.** Arms in this campaign landed up to 36%
apart in primitive count, which silently dominates every raw delta. The capacity
law is *measured* on this codebase, not borrowed: sweeping `max_points` on Garden
puts three points within 0.05 dB of **+0.5 dB per doubling of primitives**
(`cap28` predicted `−0.268`, measured `−0.270`), and it holds on the storage axis
too. Use `sota/frontier.py` and `sota/budget.py`; a result that does not clear
the frontier by more than the `±0.05` scatter is not a result.

**3. Train, do not screen.** Fifteen preregistered proxy gates produced about
four end-to-end trainings and no win. The screens are systematically pessimistic
about a jointly optimised model, and this project measured that directly
(`experiments/apx_f0/analysis_full_04.md`). A change can also pass a gradient
check and still destroy training: commit `6f7b03d` added the exact screen-space
vertex gradient to the default path and took Garden from 24.72 to **11.38 dB**,
unnoticed for six days. Only a full run tells you.

**4. Watch the compactness axis.** Their Table 2 sells 100 MB and a 2.5–15×
advantage over concurrent methods. `tex2` gained `+0.154` raw on Garden at
`−3.2%` storage, but on Room its 204.7 MB of texels had no mesh saving to hide
behind and storage went **+16.5%**. Any capacity increase must be paid for.

---

## Closed — do not reopen

| axis | how it died |
|---|---|
| Window-hardening schedule | Coverage left for the tail `0.209 / 0.111 / 0.008` → `24.72 / 24.57 / 22.79`. Monotone, and the loss is relocated not removed: the arm that emptied the tail dropped its peak from `25.30` to `23.11`. |
| Learning rate through hardening | `position_lr_max_steps` 30k → 60k costs `1.24 dB` and triples the tail loss. The decay is stabilisation, not a limitation. |
| Per-face hardening order | At a fixed softness budget, `spread` 1 / 4 / 16 → `24.72 / 24.37 / 23.86`. Unbudgeted, it degenerates: 93.6% of faces postpone everything, then collapse `25.83 → 21.47` in one step. |
| Exact vertex position gradient | Step sizes 1/10, 1/100, 1/1000 → `12.45 / 9.59 / 8.97`. Lowering the step makes it worse, so the term points somewhere harmful — most likely double counting against `BACKWARD::preprocess`, which already routes `dL_dnormals`/`dL_doffsets`/`dL_dmean2D` to vertices. |
| Densification criterion and allocator | XVR (raw residual beat coverage by 4.85%/5.53% against a locked 10%); ADC (granting `max_blending` magnitude any authority costs `0.14 dB`). |
| Sampling budget | SAC-G0/G1: scaling 4 → 2 costs `0.073 dB`, and the apparent gain was a cleanup-scaling primitive-count artefact. |
| Topology / exact subdivision | CSU, RITS-T0, RITS-P0. Exact refinement invariance is real (p99 error exactly 0) but improved neither optimisation nor quality. |
| Longer training | 30k / 50k / 90k → `24.72 / 24.88 / 24.59`, and the mid point's gain is a size effect that vanishes on Room. Confounded anyway: scaling `densify_until_iter` scales the number of pruning rounds. |

All four hardening results reduce to one statement, and any new hardening idea
must explain how it escapes it:

> **Softness has a price, it is paid the moment it is given up, and neither when
> nor on which faces you give it up changes the total.**

---

## Candidate 1 — supervise the model you actually ship

**The strongest idea in this document.** MeshSplatting minimises `L1 + DSSIM` on
the *soft* render for the whole run and only reaches the hard, opaque model on
the final update. **The model it optimises is never the model it delivers.** All
four hardening arms above changed *when* `sigma` moves; none changed *what the
loss sees*. That is why they all hit the same wall.

Compute the loss on the **hard** render while taking gradients through the soft
one — a straight-through estimator:

```python
hard = render(view, triangles, pipe, bg, sigma_override=final_sigma)["render"]
soft = render(view, triangles, pipe, bg)["render"]
image = soft + (hard - soft).detach()      # value of hard, gradient of soft
```

Anneal the mixture if the bias is too strong: `image = soft + w(t) * (hard - soft).detach()`
with `w` rising from 0 to 1 over training.

**Why it survives what killed the others.** The hardening tax is currently paid
at the end, at 2% of the initial vertex learning rate, on a model that was shaped
for a different render. Under straight-through the tax is paid continuously, at
full learning rate, and the geometry is shaped for the endpoint from the start.
This is a different intervention class from anything tried — it changes the
objective, not the schedule.

**Prior art to read before writing code**, both already in `papers/`:
`Binary Opacity Grids (SIGGRAPH 2024).pdf` (straight-through for exactly this
soft-to-binary problem) and `continuation_path_learning.pdf`. Also check whether
their `L_z` depth-alignment loss already does a weak version of this.

- **Effort**: 2–3 days. Needs a second render per iteration (~1.8× step time) or
  a kernel that emits both windows in one pass, which is the better version and
  a genuine contribution on its own.
- **Kill criterion**: if the hard-render PSNR at iteration 20k is not already
  above the baseline's *final* 24.72 on Garden, stop. That check costs 30 min.
- **Expected**: this is the only candidate I would bet on for `> +0.5 dB`, because
  it attacks the 2.4 dB directly rather than redistributing it.

---

## Candidate 2 — rebalance appearance between angle and space

Appearance is currently **100% view-dependent per-vertex and 0% view-independent
per-face**: degree-3 spherical harmonics, 16 coefficients × 3 channels = 48
floats per vertex, barycentrically interpolated in screen space. That allocation
was inherited from 3DGS, where primitives are few and large. Here there are 6.6M
faces covering roughly one pixel each, so the angular direction is almost
certainly over-provisioned and the spatial one under-provisioned. **Nobody has
ever tuned the split.**

The unfinished experiment: pay for per-face texels by lowering the SH degree.

| | floats/vertex | Room saving at 2.55M vertices |
|---|---:|---:|
| `sh_degree 3` (published) | 48 | — |
| `sh_degree 2` | 27 | 214 MB |
| `sh_degree 1` | 12 | 367 MB |

Order-2 texels cost 12 floats/face — 205 MB on Room. So `sh_degree 1 + texel_order 2`
is **net −162 MB** *and* carries more spatial detail. `sota/batch7.sh` has the
arms written (`tex2_sh2`); they were queued and cancelled to prioritise the Room
transfer test, and were never run.

- **Evidence it is live**: at a matched primitive cap on Garden, adding order-2
  texels was `+0.424 dB` over the plain capped run (`24.8755` vs `24.4517`) at 4%
  *fewer* triangles. The mechanism is right; only the payment was wrong.
- **Effort**: 1 day, all flags — the texel carrier is implemented, tested, and
  checkpointed. Sweep `sh_degree ∈ {1,2,3} × texel_order ∈ {0,2,3}` at matched
  storage on Garden, Room and Bicycle.
- **Kill criterion**: no cell clears both scenes' storage frontiers by `> 0.1 dB`.
- **Expected**: `+0.1 to +0.3 dB` at equal or lower storage. Modest but cheap,
  and it composes with everything else.

---

## Candidate 3 — make the primitive budget content-adaptive

This is the candidate that directly attacks the transfer failure. Densification
grows geometrically — `target = min(max_points, 1.23 × current)` — and stops at a
fixed iteration. It is blind to how hard the scene is, and the outcome is
absurdly inconsistent: **6.1 triangles per pixel on Garden, 3.5 on Room.** No
scene needs 6 triangles per pixel geometrically.

That inconsistency is not a curiosity, it is *why gains do not transfer*. Garden
has slack, so anything that reclaims slack helps there and nowhere else. An
allocator that equalised error per primitive would shed primitives on Garden and
add them where they are scarce — and a method whose benefit does not depend on
which scene it lands in is exactly what a nine-scene mean rewards.

**What to change.** Replace the fixed growth target with a target on a measurable
per-scene quantity — held-out error per primitive, or triangles per covered
pixel — and let the schedule run until it is met. Read `reference_code/AbsGS`
first: its homodirectional gradient is the current standard fix for "a large
blurry primitive never splits", and it is the one densification signal this
project never tested (XVR tested raw residual and coverage; ADC tested
`max_blending` magnitude; both are different quantities).

- **Effort**: 3–4 days. Touches `add_new_gs` and the pruning block in `train.py`.
- **Kill criterion**: if a content-adaptive target does not bring Garden's
  triangles-per-pixel within 30% of Room's *while holding both scenes' PSNR*,
  the premise is wrong.
- **Expected**: uncertain in absolute dB, but it is the highest-value item for
  making any other gain transfer. Worth doing before Candidate 2's sweep.

---

## Candidate 4 — filter, do not supersample

MeshSplatting renders at 4× linear resolution and area-downsamples: brute force
against aliasing. SAC-G0 measured that dropping to 2× costs only `0.073 dB`,
which says **more samples do not help** — so the residual is a *filtering*
problem, not a sampling-rate problem. At roughly one triangle per pixel with a
screen-space linear colour ramp inside each, high-frequency content (foliage in
`garden`, `bicycle`, `stump`, `treehill`) is exactly where an opaque mesh
aliases.

Read `papers/` for Mip-Splatting's 3D smoothing filter and 2D Mip filter and port
the idea: low-pass the vertex appearance by its projected footprint before
interpolation, rather than sampling the aliased signal more densely. The
per-vertex projected footprint is already computed — `scaling` / `image_size` in
the rasterizer — so the plumbing exists.

- **Effort**: 3–5 days including CUDA.
- **Kill criterion**: no SSIM/LPIPS gain on Garden *and* Bicycle at matched size.
- **Expected**: `+0.1 to +0.2 dB` PSNR but disproportionately better LPIPS and
  SSIM, which is useful because their LPIPS of 0.310 is the weakest number in
  their table and the one that "correlates best with human perception" by their
  own argument.

---

## Candidate 5 — spend the 2.4 dB where it is actually lost

The gap decomposes: opaque-only Triangle Splatting collapses to **21.05**, and
connectivity plus restricted Delaunay recovers it to **24.78**. So opacity alone
costs ~6 dB and connectivity buys back ~3.7. The residual 2.4 dB against the soft
model is concentrated where one opaque surface per pixel is simply the wrong
model: foliage, fences, hair, thin structure.

A bounded relaxation keeps the deliverable. **Allow exactly K opaque layers per
pixel** (K = 2), i.e. ship two meshes instead of one. Both are still exact
opaque triangle meshes, both still load in a game engine without custom shaders,
and the storage cost is bounded and reportable. This is the only candidate that
addresses the 2.4 dB as a *representation* question rather than an optimisation
one.

- **Effort**: 1–2 weeks; substantial CUDA work in the compositor and a real
  decision about how the two layers are assigned.
- **Kill criterion**: measure the ceiling first, cheaply — render the trained
  baseline with the *second* depth layer also composited and see how much of the
  2.4 dB a second layer could recover at all. One day. If it is under 0.5 dB,
  drop the whole candidate.
- **Expected**: the largest possible payoff here, and the largest risk of a
  reviewer saying the claim changed.

---

## Candidate 6 — initialise the connectivity from a better prior

Both stages start from COLMAP's sparse cloud. For a Gaussian method that barely
matters; for a *mesh* method it matters much more, because `run_restricted_delaunay`
builds the connectivity out of whatever the point set became. A dense prior
(MASt3R / DUSt3R, or fused monocular depth) gives the Delaunay step a far better
substrate.

Orthogonal to everything else, and cheap to try if the prior is off the shelf.

- **Effort**: 2–3 days, mostly data plumbing.
- **Kill criterion**: no improvement in the post-Delaunay PSNR at iteration 12k.
  Visible within one hour of a run.

---

## Protocol work that must happen before any of this counts

1. **Generate the Depth-Anything-V2 inverse-depth maps.** The official outdoor
   protocol in `full_eval.py` uses them (`depth_lambda_init = 0.01`); we never
   generated them, so every number in this campaign is a no-depth baseline. Their
   Table 4 says the depth loss *costs* 0.05 PSNR, so this is not a gain path — it
   is required to be comparing the same thing they compared. `README.md` has the
   commands.
2. **Run all nine scenes for the baseline, once.** 12 hours. Every future
   comparison then has a real reference instead of a Garden number.
3. **Adopt the three-scene screen** (Garden, Room, Bicycle) as the minimum before
   anything is believed, and the nine-scene run before anything is claimed.
4. **Keep `--screen_space_gradients` off.** It is the exact derivative and it
   destroys training; see the closed table above.

---

## If I had one week and one GPU

Day 1: protocol item 2 in the background while implementing Candidate 1.
Day 2: Candidate 1's 20k kill check on Garden, then Room.
Days 3–4: if it lives, Candidate 1 on three scenes at matched size. If it dies,
Candidate 3, whose kill check is also cheap.
Day 5: Candidate 2's sweep, which composes with whatever survived.
Days 6–7: nine scenes for the survivor, or write up the negative properly.

Candidate 1 first because it is the only one that changes the objective rather
than a component, and every component has now been tried.
