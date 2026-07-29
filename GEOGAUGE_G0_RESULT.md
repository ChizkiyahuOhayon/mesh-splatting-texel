# GeoGauge G0 result

Date: 2026-07-29  
Commit parent: `2bf8e00`  
Status: **FAIL — stop before scalable sketch or production integration**

## Command

```bash
bash bash_scripts/exp_geogauge_g0.sh output/geogauge_g0 cpu
```

The exact reference uses a 3×3 height field, 16 cameras, three camera baselines,
RGB/SH-1/SH-3 appearance, two surface regimes, and five seeds: 90 cases total.

Numerical checks passed:

- geometry Jacobian finite-difference error: `4.32e-12`
- appearance Jacobian finite-difference error: `3.67e-12`
- Schur symmetry error: `0`
- Schur minimum eigenvalue: `1.33e-05`

## Preregistered verdict

| Check | Threshold | Result |
|---|---:|---:|
| conditional info vs geometry error Spearman | `<= -0.70` | `-0.088` FAIL |
| absolute correlation gap vs best baseline | `>= 0.15` | `-0.757` FAIL |
| perturb prediction vs refit oracle | `>= 0.80` | `0.940` PASS |
| matched ordering | `>= 0.85` | `0.944` PASS |

The best baseline was the local residual (`rho=0.845`).  The Schur quantity
accurately predicts how much a finite geometry perturbation can be compensated by
refitting appearance, but it does not predict the geometry error left by joint
optimization.  The central paper claim therefore fails even though the local
linearization is numerically correct.

Per the v8 stop rule, do not implement G1 randomized sketches, active view
selection, or production CUDA for this direction.
