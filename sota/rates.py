"""What did the faces actually ask for?

    python -m sota.rates <run_dir> [<run_dir> ...]

Reads the learned per-face hardening rates out of a finished run and reports
their distribution. Rate 1 is the published schedule, below 1 is a face that
hardened ahead of the clock, above 1 is one that stayed soft for longer.

This is worth reading even when the metrics do not move: a distribution that
stayed at 1 says the faces did not want different clocks, while a wide one that
bought nothing says they did want them and it did not help. Those are different
results and only this tells them apart.
"""

import sys
from pathlib import Path

import torch

from sota.hardness import DEFAULT_SPREAD, rate


QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def latest_checkpoint(run):
    saves = sorted(run.glob("point_cloud/iteration_*/point_cloud_state_dict.pt"),
                   key=lambda p: int(p.parent.name.split("_")[-1]))
    if not saves:
        raise FileNotFoundError(f"no saved point cloud under {run}")
    return saves[-1]


def report(run):
    path = latest_checkpoint(run)
    state = torch.load(path, map_location="cpu")
    if not state.get("face_hardness_enabled", False):
        print(f"{run.name}: no per-face hardening rate in {path.parent.name}")
        return
    # Reconstruct with the same function training used, not the stored parameter:
    # the raw value is bounded through a tanh and normalised to mean one, so
    # exponentiating it directly reports a spread the model never rendered.
    spread = float(state.get("face_hardness_spread", DEFAULT_SPREAD))
    rates = rate(state["face_hardness"].float(), spread)
    quantiles = torch.quantile(rates, torch.tensor(QUANTILES))

    print(f"\n=== {run.name} ({path.parent.name}, {rates.numel():,} faces, "
          f"spread {spread:g}) ===")
    print("  " + "  ".join(f"p{int(q * 100):02d} {v:.4f}"
                           for q, v in zip(QUANTILES, quantiles.tolist())))
    print(f"  mean {rates.mean():.4f}   sd {rates.std():.4f}"
          f"   min {rates.min():.4f}   max {rates.max():.4f}")
    # Rate 1 is the schedule, so the split says which way the loss pushed.
    earlier = float((rates < 0.99).float().mean())
    later = float((rates > 1.01).float().mean())
    print(f"  hardened earlier than the schedule: {earlier:6.2%}")
    print(f"  stayed softer for longer:           {later:6.2%}")
    print(f"  within 1% of the schedule:          {1 - earlier - later:6.2%}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for argument in sys.argv[1:]:
        report(Path(argument))
