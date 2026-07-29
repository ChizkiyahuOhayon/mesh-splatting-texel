# FMMS G0: frozen-checkpoint native-antialiasing gate

Status: **PREREGISTERED — no G0 result has been observed**  
Date: 2026-07-29  
Research plan: FMMS-v9

## Question

Can native hard rasterization with analytic triangle-coverage antialiasing recover
the image quality of MeshSplatting's 4x-linear supersampling at materially lower
rendering cost, without changing geometry or learned appearance?

## Fixed inputs

The confirmatory gate uses the final baseline checkpoints for exactly these scenes:

1. Garden (outdoor positive-control scene from the earlier texel study)
2. Room (indoor texel-regression scene)
3. Stump (outdoor texel-regression scene)

The server-side driver must receive the dataset and model path for all three scenes.
Before rendering, it writes `g0_manifest.json` containing the resolved paths, final
iteration, SHA-256 of each `point_cloud_state_dict.pt`, image dimensions, test-view
names, package versions, GPU name, git commit, and all evaluation settings. A
confirmatory run is invalid if any input is absent or if the manifest is written
after an output image.

## Compared variants

All variants use the same checkpoint, camera, background, vertex SH evaluation,
triangle indices, and output resolution.

| ID | Rasterization | Coverage filtering | Internal resolution |
|---|---|---|---:|
| `ssaa4` | existing MeshSplatting rasterizer | area downsample | 4x linear |
| `point1` | hard z-buffer | none | 1x linear |
| `aa1` | hard z-buffer | analytic coverage AA | 1x linear |
| `aa2` | hard z-buffer | analytic coverage AA + area downsample | 2x linear |

`point1` is a negative control. `ssaa4` is the published-quality reference. No
training, geometry cleanup, opacity tuning, per-scene parameter, or learned filter
is permitted in G0.

## Metrics and timing

- Full test set: paired PSNR, SSIM, and LPIPS-VGG per view and per scene.
- Diagnostics: mean absolute pixel error against `ssaa4`, 95th percentile absolute
  error, and silhouette-band error where the `ssaa4` alpha changes spatially.
- Timing: first five lexicographically sorted test views; 5 untimed warm-ups followed
  by 20 timed repetitions per variant and view.
- Timing excludes model loading, metric calculation, image encoding, and disk I/O.
- Each timed call is enclosed by CUDA events and synchronized before aggregation.
- Report median and interquartile range; peak allocated CUDA memory is reset and
  measured separately for every variant.

## Predictions

1. `point1` will be visibly and metrically worse than `ssaa4`, reproducing the
   sampling bottleneck.
2. `aa1` or `aa2` will recover most of the difference because final-stage errors are
   concentrated at triangle coverage discontinuities.
3. `aa1`/`aa2` will reduce rendering time and peak memory because they evaluate 1/16
   or 1/4 as many raster samples as `ssaa4`.

## Precommitted pass/fail rule

G0 passes only if one analytic-AA variant satisfies **both** conditions:

1. On every scene, it is within 0.10 dB PSNR, 0.003 SSIM, and 0.005 LPIPS of
   `ssaa4`.
2. Across the three scenes, its median render time or peak allocated memory is at
   least 4x better than `ssaa4`.

The projection/color implementation must additionally pass the sanity checks below.

- Vertex clip coordinates agree with the baseline projection convention.
- The native image is not transposed or vertically/horizontally flipped.
- Background color and test-view ordering are identical.
- Metrics are computed from linear `[0,1]` tensors before PNG quantization; PNGs are
  diagnostic artifacts only.

Decision:

- **PASS:** proceed to G1 renderer-matched final refinement.
- **MIXED:** one implementation-correctness fix is allowed, then the locked protocol
  is rerun once without changing thresholds.
- **FAIL:** if both `aa1` and `aa2` fail quality on two or more scenes, or neither
  reaches 4x efficiency, stop FMMS renderer work and start exact DETRIS K=0/K=3
  reproduction.

## Confirmatory versus exploratory

Only the four variants and thresholds above are confirmatory. Alternative kernels,
orientation experiments, topology repair, learned parameters, scene subsets, or
post-hoc thresholds are exploratory and cannot be used to declare G0 passed.

