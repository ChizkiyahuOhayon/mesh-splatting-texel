"""Build the paper's tables from the archived evidence in this repository.

Every number comes from a JSON file in ``results/formal`` whose SHA-256 is
recorded in ``results/SHA256SUMS``, so a table cannot drift from the run that
produced it. Verify first, then regenerate:

    shasum -a 256 -c results/SHA256SUMS
    python results/make_tables.py [--out results/tables.tex]
"""

import json
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "formal"


def load(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def main_table(m360, tandt, deep_blending):
    lines = [
        r"\begin{tabular}{l ccc ccc ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Mip-NeRF360} & \multicolumn{3}{c}{Tanks\&Temples}"
        r" & \multicolumn{3}{c}{Deep Blending} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Method & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$"
        r" & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$"
        r" & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\",
        r"\midrule",
    ]
    rows = [
        ("MeshSplatting", "stock", "reference_means"),
        ("+ opacity floor (v1)", "ours_quality", "reference_means"),
        (r"\textbf{Ours}", "ours_quality", "means"),
    ]
    for label, arm, source in rows:
        cells = []
        for block in (m360, tandt, deep_blending):
            metrics = block[source][arm]
            cells += [fmt(metrics["psnr"], 2), fmt(metrics["ssim"]), fmt(metrics["lpips_vgg"])]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def per_scene_table(m360, equal_budget):
    """Mip-NeRF360 per scene, with the equal-budget column beside it."""
    by_scene = {row["scene"]: row for row in equal_budget["rows"]}
    lines = [
        r"\begin{tabular}{l rrr r}",
        r"\toprule",
        r"Scene & v1 & Ours & $\Delta$ & Ours @ v1's face count \\",
        r"\midrule",
    ]
    for scene in m360["scenes"]:
        ours = m360["rows"][scene]["ours_quality"]["psnr"]
        v1 = by_scene[scene]["v1_psnr"]
        matched = by_scene[scene]["matched_psnr"]
        lines.append(f"{scene} & {fmt(v1, 2)} & {fmt(ours, 2)} & {ours - v1:+.3f} & {fmt(matched, 2)} \\\\")
    v1_mean = m360["reference_means"]["ours_quality"]["psnr"]
    ours_mean = m360["means"]["ours_quality"]["psnr"]
    matched_mean = v1_mean + equal_budget["mean_psnr_gain_db"]["matched_to_v1_budget"]
    lines += [
        r"\midrule",
        f"mean & {fmt(v1_mean)} & {fmt(ours_mean)} & {ours_mean - v1_mean:+.3f} & {fmt(matched_mean)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return lines


def ablation_table(ablation):
    """Which of the two applications of the integral carries the gain."""
    means, gains = ablation["means"], ablation["psnr_gain_vs_v1"]
    wins, scenes = ablation["psnr_wins_vs_v1"], len(ablation["scenes"])
    lines = [
        r"\begin{tabular}{cc rrr r}",
        r"\toprule",
        r"Training & Cleanup & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$"
        r" & $\Delta$PSNR \\",
        r"\midrule",
    ]
    rows = [
        ("peak", "peak", "v1"),
        ("integral", "peak", "train_only"),
        (r"\textbf{integral}", r"\textbf{integral}", "full"),
    ]
    for train, cleanup, arm in rows:
        m = means[arm]
        delta = "--" if arm == "v1" else f"{gains[arm]:+.3f} ({wins[arm]}/{scenes})"
        lines.append(f"{train} & {cleanup} & {fmt(m['psnr'], 3)} & {fmt(m['ssim'], 4)} & "
                     f"{fmt(m['lpips_vgg'], 4)} & {delta} \\\\")
    three = ablation["cleanup_only_three_scene"]
    gain = three["psnr_gain_vs_peak"]
    lines += [
        r"\midrule",
        r"\multicolumn{6}{l}{\footnotesize On "
        + ", ".join(three["scenes"])
        + r" only, with the training statistic held at the peak rule:} \\",
        f"peak & integral & {fmt(three['means']['integrated']['psnr'], 3)} & "
        f"{fmt(three['means']['integrated']['ssim'], 4)} & "
        f"{fmt(three['means']['integrated']['lpips_vgg'], 4)} & "
        f"{gain['integrated']:+.3f} \\\\",
        f"peak & random & {fmt(three['means']['shuffle']['psnr'], 3)} & "
        f"{fmt(three['means']['shuffle']['ssim'], 4)} & "
        f"{fmt(three['means']['shuffle']['lpips_vgg'], 4)} & "
        f"{gain['shuffle']:+.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return lines


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "tables.tex"))
    args = parser.parse_args()

    m360 = load("softtail_nine_scene_main_table.json")
    equal_budget = load("softtail_nine_scene_equal_budget.json")
    groups = load("softtail_tandt_deep_blending_table.json")["groups"]

    lines = [
        "% Generated from results/formal/*.json (SHA-256 in results/SHA256SUMS).",
        "% Regenerate with results/make_tables.py; do not edit numbers by hand.",
        "",
    ]
    lines += main_table(m360, groups["tanks_and_temples"], groups["deep_blending"])
    lines.append("")
    lines += per_scene_table(m360, equal_budget)

    ablation_path = EVIDENCE / "softtail_nine_scene_ablation.json"
    if ablation_path.is_file():
        lines.append("")
        lines += ablation_table(load(ablation_path.name))

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    share = equal_budget["capacity_share"]
    print(f"wrote {out}; capacity share of the gain {share:.1%}")


if __name__ == "__main__":
    main()
