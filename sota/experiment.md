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

## 2026-08-20 — frozen multi-depth opacity ceiling / Room / run 01

- Status: **completed — registered depth-layer gate failed; consistent small
  quality gain observed**
- Source revision: `afaf23f76aa59343754349bf27436097555031cd`
- Output: `/home/smbu/dy/nas/meshsplatting_smbu/experiments/depth_opacity_01/room`
- Device: NVIDIA A40, logical CUDA device 0 mapped from physical GPU 1
- Environment: Python 3.11.15, PyTorch 2.7.1+cu126, CUDA toolkit 12.6
- Runtime: `92.6737976768054` seconds
- Selection: 32 fixed training views; scale `0.8` won with mean PSNR
  `31.035257935523987`, versus `30.915599048137665` at scale `1.0`
- Same-evaluator baseline test: L1 `0.022451156224959936`, PSNR
  `28.47551380059658`, SSIM `0.8746312887240679`, LPIPS
  `0.2704788966056628`, FPS `17.30245721366323`
- Selected scale `0.8` test: L1 `0.02220378894931995`, PSNR
  `28.622274447710087`, SSIM `0.8797348049970773`, LPIPS
  `0.2679430842399597`, FPS `14.939835072668084`

Against scale `1.0` in the same evaluator, scale `0.8` changes L1 by
`-0.0002473673` (`-1.102%`, better), PSNR by `+0.1467606471 dB`, SSIM by
`+0.0051035163`, LPIPS by `-0.0025358124` (`-0.938%`, better), and FPS by
`-13.655%`.  The training-view choice transfers cleanly: every registered
quality metric improves on the official test split without using a test target
for selection.

The `+0.1468 dB` test gain is nevertheless far below the registered `+0.5 dB`
ceiling threshold.  Therefore do not build a two-layer carrier or unrestricted
multi-depth representation from this result.  The narrower observation remains
actionable: the published terminal opacity is slightly too hard.  A single
stock Room training run whose only change is a terminal opacity floor of `0.8`
is the next minimal test; it asks whether optimizing under the deployment
compositor can amplify the frozen post-hoc gain.  It must be judged against a
fresh matched stock evaluation because the historical training report and this
checkpoint-reload evaluator have different timing and a small metric offset.

## Planned experiment — terminal opacity floor 0.8 / Room

Run one 30,000-iteration Room arm with the published configuration and seed.
The only training change is the endpoint of the existing opacity-floor schedule:
`0.8` instead of `0.9999`; initialization remains `0.1`, and all geometry,
capacity, loss, densification, SH, and rendering settings remain stock.  The
actual terminal floor is persisted in the checkpoint and restored on reload.
Continue only if final test PSNR and SSIM are higher and LPIPS is lower than the
matched stock Room result.  Otherwise stop without sweeping the endpoint.

## 2026-08-21 — terminal opacity floor 0.8 / Room / run 01

- Status: **completed — registered Room quality gate passed**
- Source revision: `d4bd96955f6410b91c3333acd151e3e36157e36a`
- Output: `/home/smbu/dy/nas/meshsplatting_smbu/experiments/opacity_floor_01/opacity08__room`
- Device: NVIDIA A40, logical CUDA device 0 mapped from physical GPU 1
- Environment: Python 3.11.15, PyTorch 2.7.1+cu126, CUDA toolkit 12.6
- Training time: `7,363` seconds (2 h 2 min 43 s)
- Final iteration: 30,000
- Checkpoint contract: `opacity_floor = 0.8` verified after saving
- Final test: L1 `0.022241393199715857`, PSNR `28.67998846983298`, SSIM
  `0.8787254767540174`, LPIPS `0.26390550572138566`, FPS
  `13.501303427865384`

Against the matched stock Room result, terminal opacity `0.8` changes L1 by
`+0.0000800819` (`+0.361%`, worse), PSNR by `+0.1269200154 dB`, SSIM by
`+0.0040123203`, LPIPS by `-0.0046331527` (`-1.725%`, better), and FPS by
`-56.893%`.  It passes the registered three-metric quality gate: the post-hoc
Room signal survives end-to-end training and improves PSNR, SSIM, and LPIPS
simultaneously.  It is not yet a complete SOTA result because L1 and speed
regress substantially and transfer has not been tested.

## Planned experiment — terminal opacity floor 0.8 / Garden transfer

Run the identical 30,000-iteration arm on Garden with no scene-specific method
change.  The matched stock Garden endpoint is L1 `0.0380918289689968`, PSNR
`24.721669673919678`, SSIM `0.7616264522075653`, LPIPS
`0.21650994258622328`, and FPS `32.24509159415195`.  Continue only if Garden
also improves PSNR, SSIM, and LPIPS; otherwise the Room gain is not transferable
and the direction stops without tuning the opacity endpoint.

## 2026-08-21 — terminal opacity floor 0.8 / Garden / run 01

- Status: **completed — registered Garden transfer gate passed**
- Source revision: `51106246ca08937c946ca41b608ff6daf821de93`
- Output: `/home/smbu/dy/nas/meshsplatting_smbu/experiments/opacity_floor_01/opacity08__garden`
- Device: NVIDIA A40, logical CUDA device 0 mapped from physical GPU 1
- Environment: Python 3.11.15, PyTorch 2.7.1+cu126, CUDA toolkit 12.6
- Training time: `7,557` seconds (2 h 5 min 57 s)
- Final iteration: 30,000
- Checkpoint contract: `opacity_floor = 0.8` verified after saving
- Final test: L1 `0.03679694840684533`, PSNR `25.02400803565979`, SSIM
  `0.7766205668449402`, LPIPS `0.20147701228658357`, FPS
  `12.084227031060074`

Against matched stock Garden, terminal opacity `0.8` changes L1 by
`-0.0012948806` (`-3.399%`, better), PSNR by `+0.3023383617 dB`, SSIM by
`+0.0149941146`, LPIPS by `-0.0150329303` (`-6.943%`, better), and FPS by
`-62.524%`.  Garden passes more strongly than Room and improves all four
quality metrics.  The same one-line training change has therefore transferred
across an indoor and an outdoor scene without tuning.

This is a genuine quality result, but not yet a complete SOTA result.  The
quality gain is moderate and alpha compositing processes enough additional
fragments to reduce measured FPS from `32.25` to `12.08`.  The next scientific
question is no longer whether softer terminal opacity helps; it is whether the
quality gain transfers to a third, geometrically harder scene and whether the
negligible-transmittance tail can then be skipped without losing that gain.

## Planned experiment — terminal opacity floor 0.8 / Bicycle screen

Run the identical arm on Bicycle as a third-scene transfer screen.  No matched
Bicycle run from this exact code and environment is currently recorded, so this
stage makes no causal baseline claim from external numbers.  If training is
stable and the absolute endpoint is competitive, run the unchanged stock arm
next for the matched delta; if it is clearly unusable, stop without paying for
that control.  No opacity or scene-specific setting may change.

## 2026-08-21 — terminal opacity floor 0.8 / Bicycle / run 01

- Status: **completed — stable absolute result; matched control required**
- Source revision: `39af67f392690b9643fe5828d49932960658a1d0`
- Output: `/home/smbu/dy/nas/meshsplatting_smbu/experiments/opacity_floor_01/opacity08__bicycle`
- Device: NVIDIA A40, logical CUDA device 0 mapped from physical GPU 1
- Environment: Python 3.11.15, PyTorch 2.7.1+cu126, CUDA toolkit 12.6
- Training time: `6,157` seconds (1 h 42 min 37 s)
- Final iteration: 30,000
- Checkpoint contract: `opacity_floor = 0.8` verified after saving
- Final test: L1 `0.04781450614333153`, PSNR `23.23634185791016`, SSIM
  `0.6539821124076843`, LPIPS `0.3353917896747589`, FPS
  `15.112875185900977`

The third-scene arm trains normally and produces a usable endpoint.  Because no
Bicycle baseline from the same source, environment, and metric path is recorded,
these absolute values neither pass nor fail transfer.  Do not compare them to a
paper table or another machine as if matched.

## Planned experiment — matched stock Bicycle control

Run the exact published default in the same checkout and output root: no method
flag, hence terminal opacity `0.9999`.  This single control determines all four
quality deltas and the FPS cost of `0.8` on Bicycle.  If `0.8` improves PSNR,
SSIM, and LPIPS here as on Room and Garden, proceed to inference-time tail
culling; otherwise stop the transfer claim and analyze the failure before any
speed work.

## 2026-08-21 — matched stock Bicycle control / run 01

- Status: **completed — Bicycle transfer gate passed**
- Source revision: `0190ca0fbb4cc80e3b0d6da1f084fb98d0693390`
- Output: `/home/smbu/dy/nas/meshsplatting_smbu/experiments/opacity_floor_01/stock__bicycle`
- Device: NVIDIA A40, logical CUDA device 0 mapped from physical GPU 1
- Environment: Python 3.11.15, PyTorch 2.7.1+cu126, CUDA toolkit 12.6
- Training time: `5,844` seconds (1 h 37 min 24 s)
- Final iteration: 30,000
- Checkpoint contract: `opacity_floor = 0.9999` verified after saving
- Final test: L1 `0.04879365459084511`, PSNR `23.042085647583008`, SSIM
  `0.6407600545883179`, LPIPS `0.34864730000495914`, FPS
  `16.544935566342215`

Against this matched control, opacity `0.8` changes L1 by `-0.0009791484`
(`-2.007%`, better), PSNR by `+0.1942562103 dB`, SSIM by `+0.0132220578`,
LPIPS by `-0.0132555103` (`-3.802%`, better), and FPS by `-8.656%`.
Bicycle therefore passes all four quality metrics.  Across Room, Garden, and
Bicycle, the same untuned terminal opacity improves PSNR, SSIM, and LPIPS in
every scene; L1 improves in two of three.  The quality mechanism is now
transferable enough to justify optimizing its inference cost.

## Planned experiment — frozen transmittance-tail culling / Room

Do not retrain or alter opacity.  On the trained Room `0.8` checkpoint, test
earlier termination of front-to-back compositing after the remaining
transmittance becomes negligible.  The renderer's published cutoff `1e-4`
remains the default and bitwise parent.  A fixed training-view subset may choose
among `1e-4, 3e-4, 1e-3, 3e-3, 1e-2`; choose the fastest candidate whose mean
training PSNR is within `0.02 dB` of `1e-4`, then evaluate that one choice on the
official test split.  Continue only if test PSNR loses no more than `0.03 dB`
and FPS improves by at least `25%` relative to the same checkpoint at `1e-4`.

## 2026-08-21 — frozen transmittance-tail culling / Room / run 01

- Status: **completed — stopped**
- Source revision: `d50e3af6e8a1c4980882c08ce3b67c622b063d88`
- Output: `/home/smbu/dy/nas/meshsplatting_smbu/experiments/tail_culling_01/room`
- Checkpoint: trained Room opacity-0.8 endpoint, iteration 30,000
- Default test: PSNR `28.54004820799216`, FPS `16.57662177959423`
- Selected threshold: `1e-4` (the unchanged default)

The first alternative, `3e-4`, changed the training-subset PSNR from
`30.961707592010498` to `30.49935621023178` (`-0.4623513818 dB`) while FPS
rose only from `16.40016454030132` to `16.502459001950307` (`+0.624%`).
Thresholds `1e-3`, `3e-3`, and `1e-2` reduced mean selection PSNR to
`24.1839`, `23.5632`, and `22.9343` dB respectively.  None met the locked
`0.02 dB` selection tolerance, so the default was correctly retained and the
speed gate failed.  Do not tune this cutoff further: the removed layers carry
material color, and ordinary early termination exposes the background rather
than eliminating a harmless compute-only tail.

## Planned experiment — transmittance-tail absorption / Room

Keep the same frozen Room opacity-0.8 checkpoint and cutoff grid.  For each
non-default cutoff, when the next alpha update would cross the cutoff, assign
the entire remaining transmittance to that terminal fragment and stop.  This
approximates the skipped nearby layers with the last visible surface instead of
replacing them with background.  The normal `1e-4` renderer remains the bitwise
baseline and training never uses absorption.  Select on the same fixed training
subset with the same `0.02 dB` tolerance, then test one choice.  Continue only
if test PSNR loses at most `0.03 dB` and FPS improves by at least `25%`.
