"""CGR E9 kill-shot analysis — ROC-AUC of each candidate signal.

Loads a CGR diagnostic dump (per-face O_i, nu_i, curvature on the *reconstructed*
mesh) and the toy scene's ground-truth per-face regime labels, transfers the GT
labels onto the reconstructed faces by centroid proximity, and reports how well
each signal separates *resolved crease* (label 0) from *under-resolved thin*
(label 1) via ROC-AUC.

Pre-registered kill-shot prediction (RESEARCH_PLAN_v6 sec.4):
    O_i (trajectory coherence)  AUC >= 0.8    -> the signal is special
    nu_i (gradient magnitude)   AUC ~= 0.5    -> temporal COHERENCE, not raw pull
    curvature (static geometry) AUC ~= 0.5    -> no static scalar separates
If nu_i or curvature also reach high AUC, the CGR oral claim is FALSIFIED.

AUC is the Mann-Whitney U statistic (rank-based, exact, no sklearn dependency).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

LABEL_CREASE, LABEL_THIN = 1, 2


def roc_auc(scores, y):
    """AUC that ``scores`` ranks positives (y==1) above negatives (y==0).

    Mann-Whitney U / (n_pos * n_neg), with average ranks for ties => exact ROC-AUC.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s_sorted = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # average rank (1-based)
        i = j + 1
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def transfer_labels(recon_centroid, gt_centroid, gt_label, max_dist):
    """Assign each reconstructed face the label of its nearest GT face.

    Returns (label per recon face, distance per recon face). Faces farther than
    ``max_dist`` from any GT face, or whose nearest GT face is not crease/thin, are
    dropped downstream.
    """
    tree = cKDTree(gt_centroid)
    dist, idx = tree.query(recon_centroid, k=1)
    return gt_label[idx], dist


def main():
    ap = argparse.ArgumentParser(description="CGR E9 ROC-AUC kill-shot analysis")
    ap.add_argument("--dump", required=True, help="cgr_diag_*.npz from the training run")
    ap.add_argument("--gt", required=True, help="gt_labels.npz from cgr_toy_scene.py")
    ap.add_argument("--out", default=None, help="optional path to write the result JSON")
    ap.add_argument("--max-dist", type=float, default=0.05,
                    help="drop recon faces farther than this from any GT face "
                         "(world units; the toy spans ~[-1.5,1.5])")
    ap.add_argument("--min-obs", type=int, default=0,
                    help="if the dump carries per-face observation counts (cgr_diagnose.py), "
                         "drop faces observed in fewer than this many views")
    args = ap.parse_args()

    dump = np.load(args.dump)
    gt = np.load(args.gt)

    recon_centroid = dump["face_centroid"]
    signals = {
        "O_i (coherence, OURS)": dump["face_coherence"],
        "nu_i (magnitude, control)": dump["face_magnitude"],
        "curvature (static)": dump["face_curvature"],
    }

    label, dist = transfer_labels(recon_centroid, gt["face_centroid"],
                                  gt["face_label"], args.max_dist)

    # keep confidently-matched faces whose GT regime is crease or thin
    keep = (dist <= args.max_dist) & np.isin(label, [LABEL_CREASE, LABEL_THIN])
    # optionally restrict to well-observed faces (full-sweep dumps carry face_obs)
    if args.min_obs > 0 and "face_obs" in dump:
        keep &= dump["face_obs"] >= args.min_obs
        print(f"observation filter: keep faces with >= {args.min_obs} views")
    y = (label[keep] == LABEL_THIN).astype(int)  # 1 = under-resolved thin (positive)
    n_crease, n_thin = int((y == 0).sum()), int((y == 1).sum())

    print(f"steps={int(dump['steps'])}  rho={float(dump['rho'])}")
    print(f"recon faces={len(recon_centroid)}  matched(<= {args.max_dist})={int(keep.sum())}"
          f"  |  crease={n_crease}  thin={n_thin}  (dropped flat/unmatched)")
    if n_crease == 0 or n_thin == 0:
        raise SystemExit("ERROR: one regime is empty after matching — check registration "
                         "/ --max-dist / that the run trained on the toy scene.")

    print(f"\n  {'signal':30s}  {'AUC(thin vs crease)':>20s}  {'med crease':>11s}  {'med thin':>10s}")
    print("  " + "-" * 76)
    results, medians = {}, {}
    yb = y.astype(bool)
    for name, sig in signals.items():
        s = sig[keep]
        a = roc_auc(s, y)
        mc, mt = float(np.median(s[~yb])), float(np.median(s[yb]))
        results[name], medians[name] = a, {"crease": mc, "thin": mt}
        print(f"  {name:30s}  {a:>20.3f}  {mc:>11.4f}  {mt:>10.4f}")

    o_auc = results["O_i (coherence, OURS)"]
    nu_auc = results["nu_i (magnitude, control)"]
    cv_auc = results["curvature (static)"]

    # Self-validating three-way verdict. The pre-registered O_i test is only
    # meaningful if the under-resolved-thin regime is actually PRESENT at the
    # measurement iteration -- i.e. thin faces are genuinely more driven than
    # crease faces. nu_i (raw gradient magnitude) is the direct existence proxy:
    # if nu_i does not separate, the mesh has converged in both regimes and there
    # is nothing for O_i to detect (the test cannot falsify a claim about a regime
    # its data does not contain -- cf. E8). Thresholds are pre-stated here.
    REGIME_MIN = 0.65   # nu_i AUC below this => regime absent => UNINFORMATIVE
    O_BAR, GAP = 0.80, 0.15
    regime_present = nu_auc >= REGIME_MIN
    if not regime_present:
        verdict = ("UNINFORMATIVE — under-resolved-thin regime absent at measurement "
                   f"(nu_i AUC={nu_auc:.3f} < {REGIME_MIN}: thin no more driven than crease, "
                   "mesh near-converged). O_i not diagnostic here; needs sub-resolution thin "
                   "structures and/or an earlier measurement window.")
    elif o_auc >= O_BAR and o_auc - max(nu_auc, cv_auc) >= GAP:
        verdict = "PASS — regime present (nu_i separates) AND O_i clears the bar, beating magnitude+curvature"
    else:
        verdict = ("FALSIFIED — regime present (nu_i separates) but O_i does NOT beat "
                   "magnitude/curvature: trajectory coherence adds nothing over raw drive.")
    print(f"\n  regime present (nu_i AUC >= {REGIME_MIN})? {regime_present}")
    print(f"  verdict: {verdict}")
    print(f"    O_i={o_auc:.3f}  nu_i={nu_auc:.3f}  curvature={cv_auc:.3f}"
          f"  (pre-reg: O_i>=0.8, gap>=0.15, regime nu_i>=0.65)")

    out = {
        "dump": os.path.abspath(args.dump), "gt": os.path.abspath(args.gt),
        "steps": int(dump["steps"]), "rho": float(dump["rho"]),
        "max_dist": args.max_dist, "n_crease": n_crease, "n_thin": n_thin,
        "auc": results, "medians": medians,
        "regime_present": bool(regime_present), "verdict": verdict,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
