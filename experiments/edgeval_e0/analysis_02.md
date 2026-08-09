# EdgeVal-E0 attempt 02 — sealed representation-gate result

Date observed: 2026-08-09

Decision: **PASS**

## Identity

- Source revision: `3db3e20772f5c62934a64e637b2264dc5b46ca51`
- Physical GPU: 2, NVIDIA A40, 46,068 MiB total, 45,415 MiB free
- `CUDA_VISIBLE_DEVICES`: `2` (one logical CUDA device visible)
- PyTorch: `2.7.1+cu126`
- CUDA runtime: `12.6`
- Persistent directory:
  `/home/smbu/dy/nas/meshsplatting_smbu/experiments/edgeval_e0_02`
- Native extension:
  `/home/smbu/micromamba/envs/mesh_splatting/lib/python3.11/site-packages/diff_triangle_rasterization/_C.cpython-311-x86_64-linux-gnu.so`

## Gate results

The extension wheel built and installed successfully. All 21 mathematical-core
tests passed. The fixed CUDA triangle fixture produced:

| preregistered check | result |
|---|---:|
| zero edge detail is bitwise parent | pass |
| nonzero edge detail changes render | pass |
| analytic edge gradient is finite | pass |
| analytic gradient matches central difference | pass |

The baseline and active-zero render share SHA-256
`5f322709cde086cb408aecee345fd108fdfb4d14979c5b0980a05e0f05caa2c0`.
The nonzero render has SHA-256
`8ae8c93a0e3ee16fcc8a5f4aa8518445e5e281808f4e538afc9cfbb00ab9e40f`
and differs from baseline by maximum absolute RGB `0.09898531436920166`.

Analytic gradient:

```text
[157.6490020751953, -47.294715881347656, 31.529802322387695]
```

Central-difference gradient:

```text
[157.6461639404297, -47.294612884521484, 31.528470993041992]
```

Maximum relative error is `4.22244738729205e-05`, versus the locked tolerance
`5e-3` (approximately 118 times below the limit).

## Artifact hashes

| artifact | SHA-256 |
|---|---|
| `gpu.txt` | `f9cadc2ca9ab32a5873493a6d9e46d63abf96fc1ac72ece272584b0859ae6a15` |
| `python_env.txt` | `26be5672cec427b22f53f5a3db79bb291df53bb993fa950b3d827bc2ee6728c3` |
| `build.log` | `f33c0f299b5d79af22bd8dc613f524d09459c947ce63d016f6c3bbbb5b2102ad` |
| `tests.log` | `07bf5da932e6eded82efee4aac3e29b4ea276b60ddfec14378f65f34a7a849c1` |
| `result.json` | `06f2d80ad70f7a49df709d59fe7f5784acaa68fcd8b1b890de129d87aa4bc742` |
| `smoke.log` | `06f2d80ad70f7a49df709d59fe7f5784acaa68fcd8b1b890de129d87aa4bc742` |
| `manifest.json` | `5a46473e6496d45cdea00578829bf90c034bfa0ffd0ff87bfee7254e8a4eb376` |
| `DONE` | `37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1` |
| compiled extension | `fd38a398bf67971a30a62011489774316610228a2f709dc4099e65aaefd3e967` |

`result.json` and `smoke.log` intentionally have the same hash because the
smoke program prints exactly the serialized result that it writes to disk.

## Interpretation and authorization

E0 establishes implementation correctness for the optional connected P2 edge
carrier on the fixed fixture. It does not establish Garden utility, predictor
transfer, final quality, or novelty by itself. The pass authorizes E1 protocol
freezing and implementation. It does not authorize observing Garden E1 scores
before its candidate set, exact sufficient-statistic convention, controls,
resource ceiling, and executable pass predicate are committed.
