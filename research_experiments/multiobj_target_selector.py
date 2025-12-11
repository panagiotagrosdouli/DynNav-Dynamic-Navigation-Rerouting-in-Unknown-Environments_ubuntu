import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PARETO_CSV = BASE / "multiobj_pareto_front.csv"


def load_pareto():
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


def argmin(items, key_fn):
    return min(items, key=key_fn)


def argmax(items, key_fn):
    return max(items, key=key_fn)


def select_target(strategy="risk_averse"):
    """
    Επιστρέφει (x, y) στόχο από το Pareto front.
    """
    pts = load_pareto()

    if strategy == "min_uncertainty":
        sol = argmin(pts, lambda r: r["uncertainty"])

    elif strategy == "max_coverage_per_meter":
        sol = argmax(pts, lambda r: r["coverage"] / (r["path_cost"] + 1e-6))

    elif strategy == "risk_averse":
        alpha, beta, gamma = 0.3, 1.0, 0.5
        sol = argmin(
            pts,
            lambda r: alpha * r["path_cost"] +
                      beta * r["uncertainty"] -
                      gamma * r["coverage"]
        )

    elif strategy == "risk_seeking":
        alpha, beta, gamma = 0.3, 0.2, 1.0
        sol = argmin(
            pts,
            lambda r: alpha * r["path_cost"] +
                      beta * r["uncertainty"] -
                      gamma * r["coverage"]
        )

    else:
        raise ValueError(f"Unknown strategy {strategy}")

    return int(sol["x"]), int(sol["y"]), sol
