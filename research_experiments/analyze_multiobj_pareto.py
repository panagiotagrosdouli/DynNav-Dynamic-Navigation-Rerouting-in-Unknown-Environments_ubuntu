import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent

PARETO_CSV = BASE / "multiobj_pareto_front.csv"
OUT_MD = BASE / "phd_experiments" / "multiobj_pareto_summary.md"
PLOTS_DIR = BASE / "phd_experiments" / "multiobj_pareto_plots"


def load_pareto_csv():
    """
    CSV columns:
    x, y, entropy_gain, uncertainty_gain, distance, score

    Mapping:
      coverage    := entropy_gain
      uncertainty := uncertainty_gain
      path_cost   := distance
    """
    rows = []
    with PARETO_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "x": float(r["x"]),
                "y": float(r["y"]),
                "coverage": float(r["entropy_gain"]),
                "uncertainty": float(r["uncertainty_gain"]),
                "path_cost": float(r["distance"]),
                "score": float(r["score"]),
            })
    return rows


def compute_front_curvature(points):
    pts = sorted(points, key=lambda x: x[0])
    if len(pts) < 3:
        return math.nan, None

    cov = np.array([p[0] for p in pts])
    cost = np.array([p[1] for p in pts])

    curv = []
    for i in range(1, len(cost) - 1):
        x_prev, x_curr, x_next = cov[i - 1], cov[i], cov[i + 1]
        y_prev, y_curr, y_next = cost[i - 1], cost[i], cost[i + 1]

        dx1 = x_curr - x_prev
        dx2 = x_next - x_curr
        if dx1 == 0 or dx2 == 0:
            curv.append(0.0)
            continue

        d1 = (y_curr - y_prev) / dx1
        d2 = (y_next - y_curr) / dx2
        curv.append(d2 - d1)

    curv = np.array(curv)
    mean_curv = float(np.nanmean(curv))
    knee_idx_local = int(np.argmax(curv))
    knee_idx_global = knee_idx_local + 1
    return mean_curv, knee_idx_global


def scatter_2d(xs, ys, color, title, xlabel, ylabel, fname, color_label):
    plt.figure()
    sc = plt.scatter(xs, ys, c=color, cmap="viridis", s=30, alpha=0.7)
    plt.colorbar(sc, label=color_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def main():
    if not PARETO_CSV.exists():
        print(f"[ERROR] {PARETO_CSV} not found.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_pareto_csv()

    coverage = [r["coverage"] for r in rows]
    uncertainty = [r["uncertainty"] for r in rows]
    path_cost = [r["path_cost"] for r in rows]

    # 1) Coverage vs Path Cost
    scatter_2d(
        coverage,
        path_cost,
        color=uncertainty,
        title="Coverage vs Path Cost (colored by uncertainty)",
        xlabel="coverage (entropy_gain)",
        ylabel="path cost (distance)",
        fname=PLOTS_DIR / "coverage_vs_cost_unc.png",
        color_label="uncertainty_gain",
    )

    # 2) Coverage vs Uncertainty
    scatter_2d(
        coverage,
        uncertainty,
        color=path_cost,
        title="Coverage vs Uncertainty (colored by path_cost)",
        xlabel="coverage (entropy_gain)",
        ylabel="uncertainty_gain",
        fname=PLOTS_DIR / "coverage_vs_unc_cost.png",
        color_label="path_cost",
    )

    # 3) Curvature / knee
    points_cov_cost = [(r["coverage"], r["path_cost"]) for r in rows]
    mean_curv, knee_idx = compute_front_curvature(points_cov_cost)

    knee_cov = None
    knee_cost = None
    if knee_idx is not None and 0 <= knee_idx < len(points_cov_cost):
        pts_sorted = sorted(points_cov_cost, key=lambda x: x[0])
        knee_cov, knee_cost = pts_sorted[knee_idx]

    # 4) Markdown summary
    lines = []
    lines.append("# Multi-Objective Pareto Front Analysis\n")
    lines.append("CSV source: `multiobj_pareto_front.csv`\n")
    lines.append("## Global trade-off plots\n")
    lines.append("- `multiobj_pareto_plots/coverage_vs_cost_unc.png`")
    lines.append("- `multiobj_pareto_plots/coverage_vs_unc_cost.png`\n")

    if not math.isnan(mean_curv):
        lines.append(f"- Mean curvature proxy: {mean_curv:.5f}")
    if knee_cov is not None:
        lines.append(f"- Knee point approx at coverage = {knee_cov:.3f}, path_cost = {knee_cost:.3f}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Wrote Pareto analysis summary to {OUT_MD}")


if __name__ == "__main__":
    main()
