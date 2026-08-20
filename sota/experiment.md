# SOTA experiment log

This file records completed training attempts and scientific decisions.  Metrics
are copied from sealed run artifacts; infrastructure failures are documented in
`reassessment_20260820.md` and are not counted as method results.

## 2026-08-20 — Endpoint-forward / Room / run 01

- Status: **completed — endpoint-forward falsified on Room**
- Source revision: `59625f95af8bc8e6aa6262358f024fbf1d3d7c29`
- Output: `/home/smbu/dy/nas/meshsplatting_smbu/experiments/endpoint_forward_01/endpoint__room`
- Device: NVIDIA A40, logical CUDA device 0 mapped from physical GPU 1
- Environment: Python 3.11.15, PyTorch 2.7.1+cu126, CUDA toolkit 12.6
- Rasterizer wheel SHA-256: `730090f65ee46514239304dfb7040da479eb795b1b0f947b9117d5ef32f0b195`
- Training time: 8,007 seconds (2 h 13 min 27 s)
- Final iteration: 30,000
- Reported final **train** metrics: L1 `0.062209452688694`, PSNR
  `17.24694404602051`, SSIM `0.5275115191936494`, LPIPS
  `0.5549922466278077`, FPS `13.665439204288433`
- Final **test** metrics: L1 `0.07075820949215156`, PSNR
  `16.694320727617313`, SSIM `0.5265256739579715`, LPIPS
  `0.5605809703851358`, FPS `14.40546931962366`
- Artifact SHA-256: `metrics.txt`
  `ae48322a8963dea4f340403a066759eb7cfbe3c0abdde95331a746a03718e3ab`;
  `DONE` `cea261cc16f9163bb117f6f4520d2a355c541f20007131b2cdd66f78b0dde55a`

Matched Room baseline at iteration 30,000: test L1 `0.022161311278931603`,
PSNR `28.553068454448994`, SSIM `0.874713156467829`, LPIPS
`0.2685386584355281`, FPS `31.320344514488784`.

Endpoint-forward changes versus the matched baseline: L1 `+0.0485968982`, PSNR
`-11.8587477268` dB, SSIM `-0.3481874825`, LPIPS `+0.2920423119` (worse), and
FPS `-54.006%`.  All three registered quality metrics regress by a very large
margin.  This is a decisive failure under the preregistered Room-first rule:
stop endpoint-forward, do not run Garden or Bicycle, and do not add rescue
weights, schedules, or auxiliary components to this mechanism.

## Planned experiment — SH2 + texel2 / Room

Run one unchanged Room arm with `--max_points 2800000 --texel_order 2
--sh_degree 2`.  This exchanges per-vertex angular capacity for per-face spatial
capacity using existing code.  Continue only if final test PSNR and SSIM exceed
the matched Room baseline, LPIPS is lower, and checkpoint storage does not
increase.  Otherwise stop the direction without a sweep.

## 2026-08-20 — SH2 + texel2 / Room / run 01

- Status: **completed — quality gate failed; compactness improved**
- Source revision: `36fa93c4d093d2087d962cd1008215d5a4771323`
- Output: `/home/smbu/dy/nas/meshsplatting_smbu/experiments/spatial_detail_01/tex2_sh2__room`
- Device: NVIDIA A40, logical CUDA device 0 mapped from physical GPU 1
- Environment: Python 3.11.15, PyTorch 2.7.1+cu126, CUDA toolkit 12.6
- Training time: 5,945 seconds (1 h 39 min 5 s)
- Final iteration: 30,000
- Configuration: `--max_points 2800000 --texel_order 2 --sh_degree 2`
- Reported final **train** metrics: L1 `0.017345024645328524`, PSNR
  `30.569569015502932`, SSIM `0.9150214076042176`, LPIPS
  `0.20031991302967073`, FPS `15.569512112840952`
- Final **test** metrics: L1 `0.022801179200028762`, PSNR
  `28.273298801519932`, SSIM `0.8753637014291225`, LPIPS
  `0.2697810060702837`, FPS `16.501396834212393`
- Checkpoint: `550367274` bytes; SHA-256
  `db6ac5862b08e68583423d0a09658660552e1ffe02f485773e2a4e4f43463302`
- Artifact SHA-256: `metrics.txt`
  `c6378876bdac547eeabbe55d63e7b6733c1590801ce94b9f25a2cfa617455b72`;
  `DONE` `82e44531eb2b8d229e2731f02dec4b1bbf835d88f7cffca538b381ff5bfa7bc8`

Changes versus the matched Room quality baseline: L1 `+0.0006398679`, PSNR
`-0.2797696529` dB, SSIM `+0.0006505450`, LPIPS `+0.0012423476` (worse), and
FPS `-47.314%`.  Against the sealed Room reference checkpoint
(`668222367` bytes), storage falls by `117855093` bytes (`-17.637%`).

The arm demonstrates a real compactness trade but does not improve the model:
the small SSIM increase does not offset the material PSNR loss, slightly worse
LPIPS, and large rendering-speed regression.  It fails the registered
all-metrics rule.  Do not run Garden and do not sweep this SH/texel exchange.

## Planned experiment — frozen multi-depth opacity ceiling / Room

Before changing the representation or retraining, evaluate whether deeper
fragments contain useful appearance at all.  On the sealed Room stock seed-0
checkpoint, sweep the fixed global opacity scales `1.0, 0.95, 0.9, 0.8, 0.7,
0.6, 0.5, 0.375, 0.25`.  Select one scale by mean PSNR on 32 deterministic,
evenly spaced training views, then evaluate that frozen choice on every official
test view.  Continue the depth-layer direction only for a test PSNR gain of at
least `+0.5 dB`; otherwise stop.  This diagnostic permits unrestricted deeper
alpha compositing and is therefore a ceiling, not a claimed final method.
