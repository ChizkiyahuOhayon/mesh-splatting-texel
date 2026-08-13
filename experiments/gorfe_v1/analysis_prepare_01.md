# GoRFE-V1 Prepare attempt 01

Status: **NATIVE VALIDATION FAIL; NOT A SCIENTIFIC V1 RESULT**

Attempt `_01` ran on 2026-08-13 from source
`3793a0f227a5ffdeaec8c2b9b8e0dad408d009c0` on exclusive physical GPU 0,
an NVIDIA A40 with 45,415 MiB free.  The locked environment was Python 3.11.15,
Torch 2.7.1+cu126, and CUDA build 12.6.  Both sealed iteration-30000 checkpoint
identities passed before the native build:

- Garden: `dad151eca3b5e1384496eaf5c111aded327c9710e3c4f2569b5062c8f871d5d2`;
- Room: `e214ecf386200f880432992d8721c4a4d346849bfe9f76ab8a726ab4f9d37554`.

The native wheel built successfully (build-reported SHA-256
`83919243fd031e89edcd52d613d676f5c6d08c98f33097e8e17e7f10bc8f948a`),
and all 315 repository tests passed.  The A40 native gate then passed 10 of its
11 registered checks but correctly stopped preparation because the six-output
parent identity check failed for the seven-channel auxiliary `depth` tensor.
The other five parent outputs were bitwise equal.  In particular, parent RGB
was equal with SHA-256
`8e23bcd85fac6245288ad3680c0f11d84ca085019acb4e8145f4c358ca880fb2`.
The two auxiliary hashes were
`aecce0ad790747f8f52b7bdf73f7beaa7d9095ea64e1747c2d714d05de787b20`
and `5c8f97c816af3cff38b4d51471e5b737e2b8a479f463954d50e2522f6dc8b0a2`.
The partial write-once root is
`$NAS_ROOT/experiments/gorfe_v1_prepare_01`.

All exporter-specific numerical checks passed: replay mismatch and overflow
counters were zero; count/write/forward accepted-fragment totals were all
1,216; both depth layers contributed to the same 50 reduced keys; the separated
layer designs summed exactly to the full design; duplicate-before-Gram changed
the Gram by 4.68853; sparse carrier reconstruction had maximum absolute error
`7.79e-8`; and the squared-loss gradient identity error was zero.

The failure is an existing parent-renderer undefined value, not an exporter
perturbation.  `forward.cu` declared `pixel_influence` without initialization
and wrote it into auxiliary channel 6 even when a valid image pixel accumulated
no fragment.  Two otherwise identical forwards therefore wrote different
undefined background face IDs.  Face zero is valid, so the semantically correct
no-contribution value is `-1`, matching every existing consumer's range mask.

The runner stopped before either Garden or Room scene preparation.  No target
RGB was decoded, no candidate freeze was produced, no DONE sentinel exists, and
no V1 rank or quality conclusion can be drawn.  Attempt `_01` remains immutable;
the one-line initialization repair and a native sentinel check are replayed only
under suffix `_02`.  Evaluation remains unauthorized until a complete Prepare
freeze is reviewed and committed.
