import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent

UNC_CSV = BASE / "phd_experiments" / "uncertainty_error_data.csv"
ASTAR_CSV = BASE / "phd_experiments" / "astar_phd_results.csv"
OUT_MD = BASE / "phd_experiments" / "uncertainty_expansions_summary.md"
PLOTS = BASE / "phd_experiments" / "uncertainty_expansions_plots"


def load_avg_sigma_per_instance():
    """
    Από το uncertainty_error_data.csv υπολογίζουμε mean σ για κάθε instance_id.
    """
    per_inst_sigma = defaultdict(list)

    with UNC_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            inst = int(r["instance_id"])
            sigma = float(r["sigma"])
            if np.isfinite(sigma):
                per_inst_sigma[inst].append(sigma)

    avg_sigma = {}
    for inst, sigmas in per_inst_sigma.items():
        avg_sigma[inst] = float(np.mean(sigmas))
    return avg_sigma


def load_expansions_unc_methods():
    """
    Από το astar_phd_results.csv παίρνουμε για κάθε instance_id και κάθε method=uncertainty_kX:
      - expansions
      - path_cost
      - grid_type
    """
    per_method = defaultdict(dict)  # (method -> inst -> stats)

    with ASTAR_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            method = r["method"]
            if not method.startswith("uncertainty_k"):
                continue
            if int(r["success"]) != 1:
                continue

            inst = int(r["instance_id"])
            grid = r["grid_type"]
            exp = int(r["expansions"])
            cost = float(r["path_cost"])

            per_method[method][inst] = {
                "grid_type": grid,
                "expansions": exp,
                "path_cost": cost,
            }

    return per_method


def scatter(x, y, title, xlabel, ylabel, out_path):
    plt.figure()
    plt.scatter(x, y, s=16, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    if not UNC_CSV.exists():
        print(f"[ERROR] Missing {UNC_CSV}. Run run_uncertainty_error_logging first.")
        return
    if not ASTAR_CSV.exists():
        print(f"[ERROR] Missing {ASTAR_CSV}. Run run_astar_comparison first.")
        return

    PLOTS.mkdir(parents=True, exist_ok=True)

    avg_sigma = load_avg_sigma_per_instance()
    per_method = load_expansions_unc_methods()

    lines = []
    lines.append("# Uncertainty vs Expansions Correlation\n")

    for method, inst_stats in sorted(per_method.items()):
        xs = []
        ys = []
        for inst, stats in inst_stats.items():
            if inst not in avg_sigma:
                continue
            xs.append(avg_sigma[inst])
            ys.append(stats["expansions"])

        if not xs:
            continue

        xs_arr = np.array(xs)
        ys_arr = np.array(ys)

        # correlation
        corr = float(np.corrcoef(xs_arr, ys_arr)[0, 1])

        out_plot = PLOTS / f"{method}_sigma_vs_expansions.png"
        scatter(
            xs_arr,
            ys_arr,
            title=f"{method}: expansions vs avg sigma",
            xlabel="avg sigma per instance",
            ylabel="expansions",
            out_path=out_plot,
        )

        lines.append(f"## Method: {method}")
        lines.append(f"- #instances: {len(xs_arr)}")
        lines.append(f"- corr(avg_sigma, expansions) = {corr:.4f}")
        lines.append(f"- Figure: `uncertainty_expansions_plots/{method}_sigma_vs_expansions.png`\n")

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Wrote summary to {OUT_MD}")


if __name__ == "__main__":
    main()
