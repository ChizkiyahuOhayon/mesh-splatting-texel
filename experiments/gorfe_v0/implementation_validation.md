# GoRFE-V0 implementation validation

Date: 2026-08-11

This is a development validation, not the sealed server attempt.  It used CPU
PyTorch `2.13.0`; the write-once V0 attempt must still run under the protocol's
locked `torch 2.7.1+cu126` environment.

## Results

- Full repository suite: 212 tests passed.
- Gate: 27/27 Boolean checks passed.
- DC maximum dense errors: Gram `5.55e-17`, RHS `2.78e-17`, RSS `2.22e-16`,
  held-out gain `1.11e-16`.
- SH1 maximum dense errors: Gram `1.39e-17`, RHS `1.39e-17`, RSS `2.22e-16`,
  held-out gain `6.66e-16`.
- Maximum order/chunk error: DC `4.44e-16`, SH1 `6.66e-16`.
- The deliberately wrong per-fragment Gram missed `6.60e-2` in DC and
  `2.08e-2` in SH1 on the shared-edge duplicate probe.
- Eight cameras contributed 72 fragment rows, reduced to 56 unique
  pixel-group rows.  Estimated peak temporary tensor memory was 1,320 bytes
  for DC and 2,504 bytes for SH1.

The gate preserves negative held-out gains.  This result validates the
implementation candidate only; it makes no transfer or image-quality claim.
