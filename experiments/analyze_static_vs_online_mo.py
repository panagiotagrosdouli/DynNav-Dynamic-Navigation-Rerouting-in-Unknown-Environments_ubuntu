import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
CSV_IN = BASE / "phd_experiments" / "static_vs_online_mo_stats.csv"
PLOTS = BASE / "phd_experiments" / "static_vs_online_mo_plots"
OUT_MD = BASE / "phd_experiments" / "static_vs_online_mo_summary.md"


def main():
    by_mode_exp = defaultdict(list)
    by_mode_cost = defaultdict(list)
    by_mode_succ = defaultdict(list)

    with CSV_IN.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            m = r["mode"]
            by_mode_exp[m].append(float(r["expansions"]))
            by_mode_cost[m].append(float(r["path_cost"]))
            by_mode_succ[m].append(int(r["success"]))

    modes = ["static", "online"]

    PLOTS.mkdir(parents=True, exist_ok=True)

    # Boxplot expansions
    plt.figure()
    plt.boxplot([by_mode_exp[m] for m in modes], labels=modes)
    plt.ylabel("Expansions")
    plt.title("Static vs Online MO-A* Expansions")
    plt.tight_layout()
    plt.savefig(PLOTS / "expansions.png")
    plt.close()

    # Boxplot cost
    plt.figure()
    plt.boxplot([by_mode_cost[m] for m in modes], labels=modes)
    plt.ylabel("Path Cost")
    plt.title("Static vs Online MO-A* Path Cost")
    plt.tight_layout()
    plt.savefig(PLOTS / "cost.png")
    plt.close()

    # Success rate
    rates = [100 * np.mean(by_mode_succ[m]) for m in modes]
    plt.figure()
    plt.bar(modes, rates)
    plt.ylabel("Success Rate (%)")
    plt.title("Static vs Online MO-A* Success")
    plt.tight_layout()
    plt.savefig(PLOTS / "success.png")
    plt.close()

    # Markdown summary
    lines = []
    lines.append("# Static vs Online MO-A* Statistical Comparison\n")
    lines.append("| Mode | Mean Expansions | Mean Path Cost | Success Rate |")
    lines.append("|------|------------------|----------------|---------------|")

    for m in modes:
        lines.append(
            f"| {m} | "
            f"{np.mean(by_mode_exp[m]):.2f} | "
            f"{np.mean(by_mode_cost[m]):.2f} | "
            f"{100*np.mean(by_mode_succ[m]):.1f}% |"
        )

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Written summary to {OUT_MD}")


if __name__ == "__main__":
    main()
