"""Per-face hardening order, at a fixed softness budget.

Triangle Splatting gives every triangle its own window exponent `sigma` and never
hardens; MeshSplatting collapses that into one scalar annealed to `sigma_final`,
which is what turns the soft field into an opaque connected mesh. Their Table 1
puts the two 2.4 dB apart, so the collapse is expensive -- but it is also what
makes the deliverable a mesh, and it cannot simply be undone.

Batch 1 closed the *global* schedule. Moving the window's area coverage earlier
than the published linear anneal is monotonically harmful on Garden -- `24.72`
with `0.209` of the coverage left for the last 5k iterations, `24.57` at `0.111`,
`22.79` at `0.008` -- and the loss is not removed by moving it, only relocated:
the arm that nearly emptied the tail also dropped its peak from `25.30` to
`23.11`. Softness is expressive and the published schedule already keeps it as
long as it can.

That leaves a different question. A scene is not uniformly hard to represent, so
faces that can afford opacity early and faces that need every iteration of
softness need not share a clock. Writing that as a per-face rate on the
schedule's remaining distance to the endpoint,

    sigma_face(t) = sigma_final + rate_face * (sigma_schedule(t) - sigma_final)

makes the endpoint an identity rather than a constraint: at the last iteration
the schedule *is* the endpoint, so every face lands on it whatever it learned.

**An unconstrained rate answers the wrong question.** Measured on Garden, a free
rate degenerates: 93.6% of faces chose to stay softer, the median rate reached
89.5, and the run sat at 25.83 dB one iteration before being forced onto the
endpoint and collapsing to 21.47 in a single step. The model did not reorder its
hardening, it postponed all of it -- and the 25.83 was just the soft model again,
the same effect as the 2.4 dB gap in their table.

So the rates are held to a **fixed budget**: they are bounded by `spread` in
either direction and normalised to mean one, so a face can only buy softness by
selling it to another. What is left is exactly the question worth asking -- given
the same total softness the published schedule spends, does it help to spend it
unevenly across faces?

    rate_face = spread ** tanh(raw_face) , then divided by its own mean

`raw = 0` gives every face rate one, i.e. the published schedule exactly. `tanh`
keeps the bound smooth so no face's gradient is ever clipped to zero, and the
mean normalisation is differentiable, so the budget constraint is part of the
optimisation rather than a projection applied after it.
"""

import torch


# Widest ratio a face may reach relative to the budget before normalisation.
# Four is enough to reorder the schedule substantially -- a face at the top of
# the range is still soft when one at the bottom has essentially hardened -- while
# keeping the jump onto the endpoint at the final iteration comparable to the one
# the published schedule already takes.
DEFAULT_SPREAD = 4.0


def rate(raw, spread=DEFAULT_SPREAD):
    """Per-face hardening rate: bounded, positive, and averaging one."""
    bounded = torch.pow(torch.as_tensor(spread, dtype=raw.dtype, device=raw.device),
                        torch.tanh(raw))
    return bounded / bounded.mean()


def face_sigma(raw, scheduled_sigma, final_sigma, spread=DEFAULT_SPREAD):
    """Per-face window exponent at this iteration.

    Every face reaches `final_sigma` exactly when the schedule does, because the
    term each one scales is the schedule's own remaining distance to it.
    """
    return final_sigma + rate(raw, spread) * (scheduled_sigma - final_sigma)
