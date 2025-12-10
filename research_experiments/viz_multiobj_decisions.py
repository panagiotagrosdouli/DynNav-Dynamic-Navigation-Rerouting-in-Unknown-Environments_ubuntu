import csv
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent

PARETO_CSV = BASE / "multiobj_pareto_front.csv"
DECISIONS_CSV = BASE / "phd_experiments" / "multiobj_decisions.csv"
OUT_DIR = BASE / "phd_experiments" / "multiobj_decision_viz"


def load_pareto():
    rows = []
    with PARETO_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "coverage": float(r["entropy_gain"]),
                "uncertainty": float(r["uncertainty_gain"]),
                "path_cost": float(r["distance"]),
            })
    return rows


def load_decisions():
    rows = []
    with DECISIONS_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "strategy": r["strategy"],
                "coverage": float(r["coverage"]),
                "uncertainty": float(r["uncertainty"]),
                "path_cost": float(r["path_cost"]),
            })
    return rows


def main():
    if not PARETO_CSV.exists():
        print(f"[ERROR] Missing {PARETO_CSV}")
        return
    if not DECISIONS_CSV.exists():
        print(f"[ERROR] Missing {DECISIONS_CSV}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pareto = load_pareto()
    decisions = load_decisions()

    cov_all = [p["coverage"] for p in pareto]
    cost_all = [p["path_cost"] for p in pareto]

    plt.figure(figsize=(8, 6))

    # Όλα τα Pareto σημεία
    plt.scatter(
        cov_all,
        cost_all,
        s=20,
        alpha=0.3,
        label="Pareto candidates",
    )

    # Markers & χρώματα για κάθε στρατηγική
    style = {
        "min_uncertainty":      {"marker": "o", "s": 120},
        "max_coverage_per_meter": {"marker": "s", "s": 120},
        "risk_averse_weighted": {"marker": "^", "s": 140},
        "risk_seeking_weighted": {"marker": "X", "s": 150},
    }

    for d in decisions:
        strat = d["strategy"]
        st = style.get(strat, {"marker": "*", "s": 120})
        plt.scatter(
            [d["coverage"]],
            [d["path_cost"]],
            marker=st["marker"],
            s=st["s"],
            label=strat,
        )

    plt.xlabel("Coverage (entropy_gain)")
    plt.ylabel("Path cost (distance)")
    plt.title("Multi-Objective Decision Strategies on Pareto Front")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_file = OUT_DIR / "decision_overlay.png"
    plt.savefig(out_file)
    plt.close()

    print(f"[INFO] Decision strategy overlay saved to {out_file}")


if __name__ == "__main__":
    main()
