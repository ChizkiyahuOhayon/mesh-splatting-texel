"""Plot what the two survival statistics disagree about.

``sota/survival_statistic.py`` measures, for every face of a trained mesh, the
published peak ``max_blending`` and the integral ``S_f = sum(alpha * T)``, and
samples their joint distribution. This draws two panels from that sample:

* left, the joint distribution, with the faces the two rules disagree about
  drawn on top - a face can be bright once and contribute almost nothing;
* right, how much integrated contribution each rule's survivors carry.

    python figures/make_statistic_figure.py --npz <run>/stat__room/statistic.npz \
        --json <run>/stat__room/statistic.json --out assets/softtail_statistic.png
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is fixed)

FLOOR = 1e-6


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    data = np.load(args.npz)
    report = json.loads(Path(args.json).read_text(encoding="utf-8"))
    peak = np.maximum(data["peak"], FLOOR)
    integral = np.maximum(data["integral"], FLOOR)
    keep_v1, keep_oats = data["keep_v1"], data["keep_oats"]
    only_peak = keep_v1 & ~keep_oats
    only_integral = keep_oats & ~keep_v1

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))

    left.hexbin(np.log10(peak), np.log10(integral), gridsize=70, bins="log",
                cmap="Blues", mincnt=1, linewidths=0)
    left.scatter(np.log10(peak[only_peak]), np.log10(integral[only_peak]), s=2,
                 color="#d95f02", alpha=0.35,
                 label=f"kept only by the peak rule ({report['kept_only_by_peak']['faces']:,})")
    left.scatter(np.log10(peak[only_integral]), np.log10(integral[only_integral]), s=2,
                 color="#1b9e77", alpha=0.35,
                 label=f"kept only by the integral ({report['kept_only_by_integral']['faces']:,})")
    left.set_xlabel(r"$\log_{10}$  peak  $\max\ \alpha T$  (published rule)")
    left.set_ylabel(r"$\log_{10}$  integral  $S_f=\sum \alpha T$  (ours)")
    left.set_title(f"Spearman {report['spearman_peak_vs_integral']:.3f}, "
                   f"{report['disagreement']['fraction']:.1%} of survivors differ")
    legend = left.legend(loc="lower right", frameon=True, markerscale=6, fontsize=8)
    legend.get_frame().set_linewidth(0.4)

    bins = np.linspace(np.log10(FLOOR), np.log10(integral.max()), 60)
    for mask, color, label in (
        (only_peak, "#d95f02", "kept only by the peak rule"),
        (only_integral, "#1b9e77", "kept only by the integral"),
    ):
        right.hist(np.log10(integral[mask]), bins=bins, color=color, alpha=0.6,
                   label=label)
    right.set_xlabel(r"$\log_{10}$  integrated contribution $S_f$")
    right.set_ylabel("faces (sampled)")
    right.set_title("The faces the two rules swap")
    right.legend(loc="upper left", fontsize=8, frameon=False)

    if args.title:
        figure.suptitle(args.title)
    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    print(f"wrote {out}; sampled {int(data['sampled']):,} of {int(data['faces']):,} faces")


if __name__ == "__main__":
    main()
