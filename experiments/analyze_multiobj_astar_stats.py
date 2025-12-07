import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
IN_CSV = BASE / "phd_experiments" / "multiobj_astar_stats.csv"
PLOTS = BASE / "phd_experiments" / "multiobj_astar_plots"
OUT_MD = BASE / "phd_experiments" / "multiobj_astar_stats_summary.md"


def main():
    if not IN_CSV.exists():
        print(f"[ERROR] Missing {IN_CSV}")
        return

    PLOTS.mkdir(parents=True, exist_ok=True)

    by_method_exp = defaultdict(list)
    by_method_cost = defaultdict(list)
    by_method_succ = defaultdict(list)

    with IN_CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            m = r["method"]
            by_method_exp[m].append(float(r["expansions"]))
            by_method_cost[m].append(float(r["path_cost"]))
            by_method_succ[m].append(int(r["success"]))

    methods = sorted(by_method_exp.keys())

    # ---------- BOXPLOT: EXPANSIONS ----------
    plt.figure()
    plt.boxplot([by_method_exp[m] for m in methods], labels=methods)
    plt.ylabel("A* Node Expansions")
    plt.title("Expansions Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS / "boxplot_expansions.png")
    plt.close()

    # ---------- BOXPLOT: COST ----------
    plt.figure()
    plt.boxplot([by_method_cost[m] for m in methods], labels=methods)
    plt.ylabel("Path Cost")
    plt.title("Path Cost Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS / "boxplot_cost.png")
    plt.close()

    # ---------- SUCCESS RATE ----------
    succ_rates = [100 * np.mean(by_method_succ[m]) for m in methods]

    plt.figure()
    plt.bar(methods, succ_rates)
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate per Method")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(PLOTS / "success_rate.png")
    plt.close()

    # ---------- MARKDOWN SUMMARY ----------
    lines = []
    lines.append("# Multi-Objective A* Statistical Evaluation\n")
    lines.append("| Method | Mean Expansions | Mean Cost | Success Rate |")
    lines.append("|--------|------------------|-----------|---------------|")

    for m in methods:
        lines.append(
            f"| {m} | "
            f"{np.mean(by_method_exp[m]):.2f} | "
            f"{np.mean(by_method_cost[m]):.2f} | "
            f"{100*np.mean(by_method_succ[m]):.1f}% |"
        )

    lines.append("\n## Figures")
    lines.append("- `multiobj_astar_plots/boxplot_expansions.png`")
    lines.append("- `multiobj_astar_plots/boxplot_cost.png`")
    lines.append("- `multiobj_astar_plots/success_rate.png`")

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Wrote summary to {OUT_MD}")


if __name__ == "__main__":
    main()
