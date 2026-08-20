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

## Next experiment — SH2 + texel2 / Room

Run one unchanged Room arm with `--max_points 2800000 --texel_order 2
--sh_degree 2`.  This exchanges per-vertex angular capacity for per-face spatial
capacity using existing code.  Continue only if final test PSNR and SSIM exceed
the matched Room baseline, LPIPS is lower, and checkpoint storage does not
increase.  Otherwise stop the direction without a sweep.
