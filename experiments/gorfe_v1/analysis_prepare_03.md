# GoRFE-V1 Prepare attempt 03

Status: **TARGET-FREE CAMERA-LAYOUT FAILURE; NOT A SCIENTIFIC V1 RESULT**

Attempt `_03` ran on 2026-08-13 from source
`f2c988a03f9698483fb441126811d88a1e74e98b` on exclusive physical GPU 0,
an NVIDIA A40, with Torch 2.7.1+cu126 and CUDA build 12.6.  Both sealed
iteration-30000 checkpoint hashes passed before the runner.  All 321 repository
tests passed, followed by every registered A40 native gate.  All six parent
outputs were bitwise equal; all replay mismatch and overflow counters were
zero; both depth layers summed exactly; sparse carrier reconstruction had
maximum absolute error `7.79e-8`; and the squared-loss gradient identity error
was zero.

The Revision 4 complete edge-star topology construction passed the
non-manifold edge that stopped attempt `_02`.  Preparation then stopped on the
first Garden target-free design render because its camera center is a row slice
of an inverted 4x4 transform and has non-contiguous storage.  The ordinary
renderer accepts this established camera contract by taking a contiguous copy,
but the new exporter required the input tensor itself to be contiguous and
raised `campos must be contiguous CUDA float32 with three elements`.

The failure occurred before any candidate state, candidate freeze, Room scene,
or `DONE` sentinel.  The preparation image-decoder sentinel remained active;
no target RGB, residual, loss, outcome, eligibility, or score was read.  This
attempt therefore provides native and topology engineering evidence only, not
a V1 rank decision or a MeshSplatting quality claim.  Attempt `_03` remains
immutable.  The authorized repair is limited to normalizing camera-center
storage at the exporter boundary, matching the ordinary forward contract,
with a regression test for a non-contiguous three-element camera view.
