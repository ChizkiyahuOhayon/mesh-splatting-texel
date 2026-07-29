# Exploratory renderer-ladder protocol

Status: **PREREGISTERED — ladder output not yet observed**
Date: 2026-07-30
Classification: exploratory mechanism diagnostic, not part of the confirmatory G0 gate

## Question

Does the Garden smoke quality gap come primarily from insufficient sampling of the
same renderer, or from changing renderer semantics from MeshSplatting's compositor to
an opaque hard z-buffer?

## Fixed input

Use the same official Garden iteration-30000 checkpoint and the first
lexicographically sorted held-out view used by smoke 02. No training, parameter
change, geometry cleanup, or per-variant color correction is allowed.

## Ladder

| Variant | Renderer | Internal scale | Filtering |
|---|---|---:|---|
| `splat1` | MeshSplatting compositor | 1x | none |
| `splat2` | MeshSplatting compositor | 2x | area downsample |
| `ssaa4` | MeshSplatting compositor | 4x | area downsample |
| `point1` | hard z-buffer | 1x | none |
| `aa1` | hard z-buffer | 1x | analytic AA |
| `aa2` | hard z-buffer | 2x | analytic AA + area downsample |
| `aa4` | hard z-buffer | 4x | analytic AA + area downsample |

Metrics are paired PSNR, SSIM, LPIPS-VGG, and image error against `ssaa4`. Timing is
diagnostic only in this one-view run.

## Precommitted interpretation

- If `aa4` remains more than 0.20 dB or 0.01 LPIPS behind `ssaa4`, classify the
  residual as **renderer-semantic mismatch**. Stop native-AA-as-replacement and do
  not run G1 in its current form.
- If `aa4` is within both tolerances but `aa1`/`aa2` remain poor, classify the
  residual as **insufficient footprint integration**. Continue only with a stronger
  coverage integral that has a plausible sub-4x cost.
- If `splat1`→`splat2`→`ssaa4` gains are small while hard-render variants remain
  poor, the original −0.8 dB ablation is not explained by ordinary edge sampling;
  revisit the baseline renderer semantics.
- If `aa4` matches but its cost is no better than `ssaa4`, quality is technically
  recoverable but the current Oral-level efficiency thesis is unsupported.

No result from this ladder can by itself declare G0 passed.
