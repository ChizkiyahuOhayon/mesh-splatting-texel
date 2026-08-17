"""Per-face hardening rate: when each triangle spends the window it has to spend.

Triangle Splatting gives every triangle its own window exponent `sigma` and never
hardens; MeshSplatting collapses that into one scalar annealed to `sigma_final`,
which is what turns the soft field into an opaque connected mesh. Their Table 1
puts the two 2.4 dB apart, so the collapse is expensive -- but it is also what
makes the deliverable a mesh, and it cannot simply be undone.

Batch 1 measured what the global schedule can and cannot do. Moving the window's
area coverage earlier than the published linear anneal is monotonically harmful
on Garden -- `24.72` with `0.209` of the coverage left for the last 5k
iterations, `24.57` at `0.111`, `22.79` at `0.008` -- and the loss is not
removed by moving it, only relocated: the arm that nearly emptied the tail also
dropped its peak from `25.30` to `23.11`. Softness is expressive, every unit of
coverage spent costs quality whenever it is spent, and the published schedule is
already near-optimal in that family because it stays soft as long as it can.

That closes the *global* schedule but not the per-face question, which is a
different claim: a scene is not uniformly hard to represent, so the faces that
can afford opacity early and the ones that need every iteration of softness need
not be on the same clock. This carrier interpolates the published schedule per
face,

    sigma_face(t) = sigma_final + rate_face * (sigma_schedule(t) - sigma_final)

with `rate_face > 0` learned. Its properties:

* **The endpoint is an identity, not a constraint.** At the last iteration
  `sigma_schedule` equals `sigma_final`, so `sigma_face` equals it too for every
  face and every rate. Nothing has to be clamped, projected, or regularised, and
  the model that comes out is the same opaque connected mesh.
* **The default is recovered exactly.** `rate_face = 1` is the published path,
  so a run starts byte-identical to the baseline.
* **Both directions are available.** `rate < 1` hardens a face ahead of the
  clock, `rate > 1` keeps it softer for longer. Batch 1 says the second is the
  direction the data wants, and an earlier one-sided version of this file
  offered only the first -- it could not have expressed the answer.
* **Gradients never die.** `d sigma_face / d rate = sigma_schedule(t) -
  sigma_final >= 0`, vanishing only at the final iteration, where nothing is
  left to decide.

`rate_face` is stored through an exponential, matching how `TriangleModel` stores
sigma itself, so it stays positive by construction and its step size is relative.
"""

import torch


def raw_from_rate(rate):
    """Stored parameter for a given hardening rate; `rate = 1` is the schedule."""
    return float(torch.log(torch.tensor(float(rate))))


def rate(raw):
    """Per-face hardening rate, positive by construction."""
    return torch.exp(raw)


def face_sigma(raw, scheduled_sigma, final_sigma):
    """Per-face window exponent at this iteration.

    Every face reaches `final_sigma` exactly when the schedule does, whatever it
    learned, because the term it scales is the schedule's own distance to that
    endpoint.
    """
    return final_sigma + rate(raw) * (scheduled_sigma - final_sigma)
