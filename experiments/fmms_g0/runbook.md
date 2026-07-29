# FMMS G0 server runbook

The protocol is locked in `protocol.md`. These commands run it without changing the
precommitted settings.

## Install the one additional dependency

```bash
pip install -r requirements-g0.txt --no-build-isolation
python -c "import nvdiffrast.torch, diff_triangle_rasterization"
```

The nvdiffrast dependency is pinned to the official NVIDIA commit inspected during
the v9 literature/code audit.

## Launch the three-scene gate

Use final **baseline** model directories. Each must contain
`point_cloud/iteration_30000/point_cloud_state_dict.pt` (or another final iteration)
and its original `cfg_args`.

```bash
nohup bash bash_scripts/exp_fmms_g0.sh \
  output/fmms_g0 0 \
  data/360_v2/garden output/baseline/garden \
  data/360_v2/room output/baseline/room \
  data/360_v2/stump output/baseline/stump \
  > output/fmms_g0_driver.log 2>&1 &
```

The paths above illustrate the required argument order; replace them with the actual
resolved server paths. The driver refuses non-empty per-scene output directories, so
an interrupted run cannot silently mix old and new images.

Before the full gate, run one exploratory Garden view to validate CUDA compilation,
projection orientation, and image writing:

```bash
python g0_native_eval.py \
  -s data/360_v2/garden -m output/baseline/garden --eval \
  --g0_scene garden --g0_output output/fmms_g0_smoke_garden \
  --g0_max_views 1 --g0_warmup 0 --g0_timing_repeats 1 --g0_timing_views 1
```

This smoke run is recorded as `confirmatory_settings: false` and cannot pass G0.

If the smoke quality is far outside the G0 tolerance, run the preregistered
single-view renderer ladder before any full-scene evaluation:

```bash
python g0_native_eval.py \
  -s data/360_v2/garden -m output/baseline/garden --eval \
  --g0_scene garden --g0_output output/fmms_g0_renderer_ladder \
  --g0_renderer_ladder --g0_max_views 1 \
  --g0_warmup 0 --g0_timing_repeats 1 --g0_timing_views 1
```

The ladder adds `splat1`, `splat2`, and `aa4`; its interpretation is locked in
`renderer_ladder_protocol.md`. It is exploratory and cannot pass G0.

## Monitor and inspect

```bash
tail -f output/fmms_g0_driver.log
cat output/fmms_g0/decision.json
```

Every scene directory contains:

- `g0_manifest.json`: checkpoint hash, environment, cameras, and locked settings;
- `results.json`: per-view and mean PSNR/SSIM/LPIPS plus boundary diagnostics;
- `timing.json`: raw timing samples and peak incremental CUDA memory;
- `ssaa4/`, `point1/`, `aa1/`, `aa2/`: diagnostic render images.

Do not declare G0 passed from a single scene. Only `g0_decide.py` applies the locked
three-scene rule.
