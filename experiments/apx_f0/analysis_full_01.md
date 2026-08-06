# APX-F0 result: INVALID — the gate measured the wrong model class

Date: 2026-08-06
Protocol: `experiments/apx_f0/protocol.md` (preregistered, commit `9838324`)
Records: `apx_f0_garden_02`, `apx_f0_room_02`
Checkpoints: `sac_g1_{garden,room}_stock_seed0_train`, iteration 30000

## Printed

```
garden  eligible 41,669 / 6,952,816  (0.60%)
        ceiling o1 -4.3810  o2 -3.8621  o4 -3.6511 dB
        concentration top10% 1.0000
        residual_mass capture 0.6682 lift 6.681 | max_blending 0.1472 | coverage 0.1996 | area 0.3850
        ceiling FAIL | concentration PASS | predictability PASS

room    eligible 62,945 / 5,628,158  (1.12%)
        ceiling o1 -4.2070  o2 -4.1196  o4 -4.1366 dB
        concentration top10% 0.9942
        residual_mass capture 0.3823 lift 3.823 | max_blending 0.1122 | coverage 0.2682 | area 0.2541
        ceiling FAIL | concentration PASS | predictability PASS
```

## Why none of that is a result

`apx_f0_eval.py` fits each cell's colour to the **target** and scores that fit against
the current model. The deployed texel carrier is **additive on top of SH**
(`cuda_rasterizer/forward.cu:775-792`, whose own comment states it):

```c
// Per-face texel carrier: an ADDITIVE residual on top of the
// ... SH carrying view-dependence; texels carry high spatial frequency.
interp += texels[texel_base + ch];
```

The gate therefore measured *replacing* the entire appearance with a static per-cell
colour. That discards SH's view dependence, which is why it loses about 4 dB — a
correct measurement of a model class nobody proposed. **It did not measure adding
capacity on top of SH, which is the hypothesis.**

The protocol carries the same error in words: *"one colour per cell, shared across all
views, which is what a texel carrier is."* That sentence is wrong and must be corrected
with the code.

`concentration` and `predictability` are void for a consequent reason: `gain` is
`clamp_min(0)` of a quantity that was negative almost everywhere, so nearly every face
scored exactly zero and the statistics describe a degenerate vector. `concentration
1.0000` on Garden is that degeneracy, not concentration.

## The repair

1. `_fit` accumulates `target - prediction`, not `target`.
2. `_score` compares `current` against `((prediction + fitted_residual) - target)^2`.
3. Eligibility is computed **per order** rather than once at the largest; an order-2
   grid needs 16 training pixels and an order-1 grid needs 4, not 64.

**This is a correctness repair, not a threshold relaxation.** The locked thresholds
(`0.30 dB` ceiling, `0.50` concentration, XVR-G0's `1.75` lift and `10%` control
margin) are unchanged, and the rerun writes to a new suffix.

## The one finding that survives

Model-class independent, and it stands:

| scene | faces | with >= 64 training pixels |
|---|---:|---:|
| Garden | 6,952,816 | **41,669 (0.60%)** |
| Room | 5,628,158 | **62,945 (1.12%)** |

At the deployed mesh density most faces are sub-pixel, so an order-4 per-face grid can
reach under 2% of the model. Lower orders will qualify far more faces and must be
re-measured, but "how much of the model can this scheme even touch" belongs in the
headline of any per-face proposal, not in its appendix.

## Process note

The bug is the same class as the finite-difference trilogy: a measurement that looked
like a decisive negative about the hypothesis while actually describing something else.
The guard that works is the one `adc_probe.py` uses — put the primitive in a torch-only
module and test it against a case whose answer is known independently. `apx_cells.py`
has that structure and its 28 tests pass; they verify the *arithmetic* of cell fitting,
which is correct. What no unit test could catch is that the arithmetic was pointed at
the wrong quantity, and the only defence against that is reading the kernel before
writing the protocol.
