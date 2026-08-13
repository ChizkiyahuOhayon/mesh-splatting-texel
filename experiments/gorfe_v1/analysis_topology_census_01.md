# GoRFE-V1 target-free topology census 01

Status: **SEALED TARGET-FREE DOMAIN EVIDENCE; NO SCIENTIFIC V1 RESULT**

The checkpoint-only census ran from source
`190fd90c909f2c17e686247628d94e4f0394d979`.  It loaded only the two sealed
iteration-30000 checkpoint tensors and their SHA-256 identities.  It read no
COLMAP camera metadata, opened no image path, decoded no RGB, and accessed no
residual, loss, outcome, eligibility, or score.  The write-once artifact is
`/home/smbu/dy/nas/meshsplatting_smbu/experiments/gorfe_v1_topology_census_01.json`
with SHA-256
`6bdd7fb456eb91da0764fdad5b247582d8eb90489993dc82412e81ede89380c5`.

| quantity | Garden | Room |
|---|---:|---:|
| checkpoint SHA-256 | `dad151ec...71d5d2` | `e214ecf3...37554` |
| vertices | 3,254,576 | 2,563,186 |
| faces | 6,952,816 | 5,628,158 |
| canonical edges | 11,165,824 | 9,345,153 |
| incidence 1 | 4,547,593 | 4,231,379 |
| incidence 2 | 4,375,561 | 3,347,984 |
| incidence >2 | 2,242,670 | 1,765,790 |
| non-manifold edge fraction | 0.200851 | 0.188952 |
| face-local slots on incidence >2 edges | 7,559,733 | 5,957,127 |
| fraction of all face-local slots | 0.362430 | 0.352817 |
| maximum incidence | 9 | 10 |
| repeated-vertex faces | 0 | 0 |
| non-manifold edges with a duplicate incident face | 0 | 0 |

Garden's incidence histogram for 3 through 9 is
`1,585,590 / 512,350 / 119,051 / 21,919 / 3,328 / 390 / 42`.
Room's incidence histogram for 3 through 10 is
`1,242,789 / 408,469 / 95,116 / 16,901 / 2,248 / 244 / 20 / 3`.
All 2,242,670 Garden and 1,765,790 Room non-manifold edges have distinct
incident faces.

The census falsifies the original manifold-only input assumption.  Removing
all incidence-greater-than-two edges would discard more than one third of the
face-local edge slots in both scenes, so exclusion cannot be described as a
rare-corruption policy.  The target-free correction is to model the renderer's
face set as a non-manifold simplicial 2-complex and share one P2 coefficient on
the complete star of each canonical endpoint pair.  The P2 restriction to the
edge is the same `4t(1-t)` from every incident face, so this is a well-defined C0
trace on a branching edge.  Candidate hashing, active coefficient cost, support,
GCV, controls, budgets, and decisions remain unchanged.

This artifact authorizes only protocol revision 4 and its tested
implementation.  It does not authorize target decode, V1 evaluation, training,
or a MeshSplatting quality claim.
