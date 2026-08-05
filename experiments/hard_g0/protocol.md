# HARD-G0: does the hardening schedule leave quality on the table?

Status: **PREREGISTERED — no HARD-G0 output has been observed**

Date: 2026-08-05

## Question

Both SAC-G0 arms lose held-out quality over their final iterations at a
constant supersampling factor: the stock arm falls `0.344 dB` between iteration
25,000 and 30,000 at `scaling 4`, and the `splat2` arm falls `0.641 dB` from its
peak at 21,000 to 30,000 at `scaling 2`. Training PSNR falls with it — stock
goes from `25.863` to `25.240` — so the train/test gap narrows and this is not
overfitting. The model is genuinely losing capacity.

That loss is the price of hardening: `sigma` anneals from `1.0` toward `1e-4`,
turning the power-law window into an indicator, and the opacity floor is pushed
to `0.9999`. A hard opaque mesh is the representation's central claim, so the
endpoint is not negotiable. The path to it might be.

Reading the two schedules side by side exposes an asymmetry:

| Quantity | Reaches its final value | Iterations left to adapt |
|---|---:|---:|
| opacity floor | 24,000 (`final_opacity_iter`) | 6,000 |
| `sigma` | 30,000 (`sigma_until`) | **0** |

Opacity is given six thousand iterations to settle at its deployed value.
`sigma` reaches its deployed value on the last step of training, so the model
never optimises at the hardness it is shipped with. HARD-G0 asks whether that
costs quality:

> If `sigma` reaches its final value 5,000 iterations early — an adaptation
> window comparable to the one opacity already receives — does the final model
> improve, at identical deployed hardness?

`sigma_until` is already an exposed parameter and the schedule clamps after it,
so this requires no code change and both arms end at exactly `sigma = 1e-4`.

## Arms

Garden, seed 0, everything identical except one value.

| Arm | `sigma_until` |
|---|---|
| stock | 30,000 (published) |
| early | 25,000 |

Both are evaluated at `scaling 2` and `scaling 4` with `sac_eval.py`, which also
records the final `sigma` so hardness parity is verified rather than assumed.

## This is a screen, not a decision

One scene and one seed cannot decide anything; SAC-G0 is the standing lesson.
The screen threshold is set from a measured quantity: SAC-G1 put Garden's
held-out PSNR seed standard deviation at `0.0157 dB`, so a difference of
`0.10 dB` is about 6.4 standard deviations and is safe to act on as a screen.

**Proceed** to a confirmation only if all hold:

1. both arms report a final `sigma` of `1e-4`, so the deployed hardness is
   identical;
2. `early@4` exceeds `stock@4` by at least `0.10 dB`;
3. LPIPS at `scaling 4` is not worse for `early`.

**Stop** otherwise. A failure ends the technical programme on this baseline;
the project then writes up what it has established rather than attempting a
twelfth variant.

If the screen passes, HARD-G1 replicates it exactly as SAC-G1 did — Garden and
Room, three shared seeds, paired differences, and an effect required to exceed
twice its own standard error — before any claim is made.

## Honest limitation of the design

Moving `sigma_until` earlier both grants an adaptation window and hardens faster
throughout training, so a positive result would not by itself separate those two
causes. The screen asks only whether the published schedule leaves quality on
the table. If it does, isolating which half is responsible is the first thing
HARD-G1 must add, by including an arm that reaches final `sigma` early along a
schedule matched in rate rather than in endpoint.

## Recorded

Final `sigma`, both scaling cells, primitive counts, training wall clock, and
the per-1000-iteration test PSNR trace from each training log, which is what
made the decline visible in the first place.
