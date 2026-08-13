# GoRFE-V1 Prepare attempt 02

Status: **TARGET-FREE TOPOLOGY INVALID; NOT A SCIENTIFIC V1 RESULT**

Attempt `_02` ran on 2026-08-13 from source
`ddef8225aba8c663b4509305fed953cb4f0b89f1` on exclusive physical GPU 0,
an NVIDIA A40. It used Python 3.11.15, Torch 2.7.1+cu126, and CUDA build
12.6. Both iteration-30000 checkpoint hashes were verified before the runner:

- Garden: `dad151eca3b5e1384496eaf5c111aded327c9710e3c4f2569b5062c8f871d5d2`;
- Room: `e214ecf386200f880432992d8721c4a4d346849bfe9f76ab8a726ab4f9d37554`.

The repaired A40 native gate **passed every registered check** after all 315
repository tests passed. All six ordinary/export parent outputs were bitwise
equal, including auxiliary `depth` hash
`681e6ac9414f7174ea1938a4881b581b269bb8cb6ae15dd6a0b4a97f3b918cfd`.
The background face-ID sentinel check passed. All replay mismatch/overflow
counters were zero; accepted-fragment count/write/forward totals were 1,216;
both depth layers contributed to all 50 shared reduced keys and summed exactly;
the duplicate-aware Gram differed from the deliberately wrong fragment Gram by
4.68853; carrier reconstruction maximum absolute error was `7.79e-8`; and the
squared-loss gradient identity error was zero.

Preparation then stopped during Garden's target-free topology construction.
The sealed mesh contains canonical edge `(0, 999916)` with three incident
faces, while the frozen V1 representation declares every edge with more than
two incident faces invalid. The exception occurred before target-free camera
rendering, eligibility construction, candidate-state sealing, Room topology,
or any target RGB decode. No candidate freeze or DONE sentinel exists, so no
V1 rank, selector, or MeshSplatting quality conclusion is available.

Attempt `_02` is immutable. A separate checkpoint-only census must quantify
non-manifold incidence in both scenes and distinguish genuine junctions from
duplicate triangles or a topology-construction defect. Any changed treatment
must be justified from that target-free evidence, frozen in a new protocol
revision, and implemented with new tests before a fresh suffix. Silently
filtering the edge or continuing `_02` is forbidden; evaluation remains
unauthorized.
