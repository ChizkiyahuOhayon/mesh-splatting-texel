# GoRFE-D0-B implementation record

Status: **MATHEMATICAL CORE IMPLEMENTED; REAL-SCENE REPLAY NOT YET RUN**

The protocol and falsifiable reading were frozen and pushed first at
`d46cb2e006a427f0794d129ee63412bf9cf03aa5`.  The implementation does not alter
that commit.

`gorfe_d0_b.py` is the minimal CPU float64 oracle.  It reuses V1's exact
duplicate-safe `(pixel, canonical edge)` reduction, keeps DC and SH1 as distinct
costed groups on a shared edge, and returns the linear term, independent-group
diagonal quadratic, simultaneous joint quadratic, additive gain, joint gain,
and signed interaction penalty.  It also applies the frozen per-scene/family
and shared-family mechanism reading.

The directed tests cover blank and single-group designs, disjoint support,
destructive and synergistic overlap, fragment duplication, DC+SH1 on one edge,
row/camera ordering, an independent scalar multi-channel oracle, additive-state
identity, normalization, every control comparison, excess interaction growth,
and the shared-family requirement.

This commit deliberately does not yet duplicate V1's scene loader, native
installer, or GPU runner.  The next implementation layer may only stream
hash-verified V1 inputs through this oracle; it must not change the frozen
selection or reading.
