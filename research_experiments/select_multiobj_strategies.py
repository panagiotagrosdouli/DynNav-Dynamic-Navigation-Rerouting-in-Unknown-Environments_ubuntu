import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PARETO_CSV = BASE / "multiobj_pareto_front.csv"
OUT_CSV = BASE / "phd_experiments" / "multiobj_decisions.csv"
OUT_MD = BASE / "phd_experiments" / "multiobj_decisions_summary.md"


def load_pareto():
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


def argmin(items, key_fn):
    best = None
    best_val = None
    for it in items:
        v = key_fn(it)
        if best is None or v < best_val:
            best = it
            best_val = v
    return best


def argmax(items, key_fn):
    best = None
    best_val = None
    for it in items:
        v = key_fn(it)
        if best is None or v > best_val:
            best = it
            best_val = v
    return best


def main():
    if not PARETO_CSV.exists():
        print(f"[ERROR] {PARETO_CSV} not found.")
        return

    pts = load_pareto()
    if not pts:
        print("[ERROR] No points in Pareto CSV.")
        return

    decisions = []

    # 1) Min uncertainty (risk-averse)
    sol_min_unc = argmin(pts, lambda r: r["uncertainty"])
    decisions.append({"strategy": "min_uncertainty", **sol_min_unc})

    # 2) Max coverage per meter (entropy_gain / distance)
    sol_cov_per_cost = argmax(
        pts,
        lambda r: (r["coverage"] / r["path_cost"]) if r["path_cost"] > 0 else -1e9,
    )
    decisions.append({"strategy": "max_coverage_per_meter", **sol_cov_per_cost})

    # 3) Risk-averse weighted sum
    alpha = 0.3   # path_cost weight
    beta = 1.0    # uncertainty weight (dominant)
    gamma = 0.5   # coverage weight
    def J_risk_averse(r):
        return alpha * r["path_cost"] + beta * r["uncertainty"] - gamma * r["coverage"]

    sol_risk_averse = argmin(pts, J_risk_averse)
    decisions.append({"strategy": "risk_averse_weighted", **sol_risk_averse})

    # 4) Risk-seeking weighted sum
    alpha2 = 0.3
    beta2 = 0.2
    gamma2 = 1.0   # dominant weight on coverage
    def J_risk_seeking(r):
        return alpha2 * r["path_cost"] + beta2 * r["uncertainty"] - gamma2 * r["coverage"]

    sol_risk_seeking = argmin(pts, J_risk_seeking)
    decisions.append({"strategy": "risk_seeking_weighted", **sol_risk_seeking})

    # ============================
    # Write CSV
    # ============================
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy",
        "x",
        "y",
        "coverage",
        "uncertainty",
        "path_cost",
        "score",
    ]

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in decisions:
            writer.writerow({
                "strategy": d["strategy"],
                "x": d["x"],
                "y": d["y"],
                "coverage": d["coverage"],
                "uncertainty": d["uncertainty"],
                "path_cost": d["path_cost"],
                "score": d["score"],
            })

    # ============================
    # Write Markdown summary
    # ============================
    lines = []
    lines.append("# Multi-Objective Decision Strategies\n")
    lines.append("CSV source: `multiobj_pareto_front.csv`\n")
    lines.append("| Strategy | x | y | coverage (entropy_gain) | uncertainty_gain | path_cost (distance) | score |")
    lines.append("|----------|---|---|--------------------------|------------------|----------------------|-------|")

    for d in decisions:
        lines.append(
            f"| {d['strategy']} | {d['x']:.2f} | {d['y']:.2f} | "
            f"{d['coverage']:.3f} | {d['uncertainty']:.3f} | "
            f"{d['path_cost']:.3f} | {d['score']:.3f} |"
        )

    OUT_MD.write_text("\n".join(lines))

    print(f"[INFO] Wrote decisions CSV to {OUT_CSV}")
    print(f"[INFO] Wrote decisions summary to {OUT_MD}")


if __name__ == "__main__":
    main()
