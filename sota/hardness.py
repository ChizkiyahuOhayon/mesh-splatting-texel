"""Per-face window hardness that may run ahead of the global schedule.

Triangle Splatting gives every triangle its own window exponent `sigma` and
never hardens; MeshSplatting collapses that into one scalar annealed to `1e-4`,
which is what turns the soft field into an opaque connected mesh. Their Table 1
puts the two 2.4 dB apart, so the collapse is expensive -- but it is also what
makes the deliverable a mesh, and it cannot simply be undone.

The compromise here is to keep the schedule as a floor on hardness and let each
face add to it. Working in hardness `h = 1 / sigma` rather than in `sigma`,

    h_face = a_face + h_schedule(t) ,    a_face >= 0

so that

    sigma_face(t) = 1 / (a_face + 1 / sigma_schedule(t)) <= sigma_schedule(t) .

Three properties follow, and they are the reason for this form:

* **The endpoint is exact.** `a_face >= 0` is precisely the condition
  `sigma_face(T) <= sigma_schedule(T)`, so every face ends at least as opaque as
  the published model. The mesh that comes out is the same kind of object.
* **The default is recovered exactly.** `a_face = 0` gives `sigma_schedule`
  back, so a zero-initialised carrier starts the run byte-identical.
* **Gradients never die.** `d sigma_face / d a_face = -sigma_face ** 2` is
  non-zero everywhere, unlike the `min` of the two, which would freeze any face
  the schedule currently binds.

The direction is one-sided on purpose: a face may harden early, never late. The
published anneal moves 31% of the window's area coverage after iteration 25k,
where the vertex learning rate has decayed to ~2% of its initial value
(`sota/sigma_schedule.py`). Letting a face spend that change earlier, while the
geometry can still respond, is the whole point; letting it defer the change
would only push more of it into the tail.
"""

import torch


# `a_face` is stored through an exponential, matching how `TriangleModel` stores
# sigma itself, so the parameter is scale-free and stays positive by
# construction. The initial value is small enough to leave the render visually
# unchanged at allocation and large enough for Adam to move it immediately.
INITIAL_ADDED_HARDNESS = 1e-3


def raw_from_hardness(added_hardness):
    return float(torch.log(torch.tensor(float(added_hardness))))


def added_hardness(raw):
    """The non-negative hardness each face adds to the scheduled floor."""
    return torch.exp(raw)


def face_sigma(raw, scheduled_sigma):
    """Per-face window exponent, never softer than the schedule.

    `scheduled_sigma` is the scalar the published path would have used at this
    iteration; the result is elementwise `<=` it.
    """
    return torch.reciprocal(added_hardness(raw) + 1.0 / scheduled_sigma)
