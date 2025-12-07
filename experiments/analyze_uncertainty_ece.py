import csv
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
ERR_CSV = BASE / "phd_experiments" / "uncertainty_error_data.csv"
OUT_MD = BASE / "phd_experiments" / "uncertainty_ece_summary.md"


def load_data():
    rows = []
    with ERR_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            h_star = float(r["h_star"])
            mu = float(r["mu"])
            sigma = float(r["sigma"])
            rows.append((h_star, mu, sigma))
    return rows


def compute_nll(rows):
    """
    Negative log-likelihood για Gaussian N(mu, sigma^2).
    NLL = 0.5*log(2πσ^2) + (h* - μ)^2 / (2σ^2)
    """
    nlls = []
    for h_star, mu, sigma in rows:
        if sigma <= 0 or not math.isfinite(sigma):
            continue
        var = sigma ** 2
        nll = 0.5 * math.log(2 * math.pi * var) + (h_star - mu) ** 2 / (2 * var)
        if math.isfinite(nll):
            nlls.append(nll)
    if not nlls:
        return float("nan")
    return sum(nlls) / len(nlls)


def compute_coverage(rows, k):
    """
    Coverage για εύρος μ ± kσ.
    Επιστρέφει το ποσοστό δειγμάτων που πέφτει μέσα σε αυτό το interval.
    """
    total = 0
    inside = 0
    for h_star, mu, sigma in rows:
        if sigma <= 0 or not math.isfinite(sigma):
            continue
        total += 1
        if abs(h_star - mu) <= k * sigma:
            inside += 1
    if total == 0:
        return float("nan")
    return inside / total


def main():
    if not ERR_CSV.exists():
        print(f"[ERROR] Missing {ERR_CSV}. Run run_uncertainty_error_logging first.")
        return

    rows = load_data()
    print(f"[INFO] Loaded {len(rows)} samples")

    mean_nll = compute_nll(rows)

    cov_1 = compute_coverage(rows, 1.0)
    cov_2 = compute_coverage(rows, 2.0)
    cov_3 = compute_coverage(rows, 3.0)

    # expected ideal Gaussian coverages
    ideal_1 = 0.6827
    ideal_2 = 0.9545
    ideal_3 = 0.9973

    lines = []
    lines.append("# Uncertainty Calibration Metrics (Regression-style)\n")
    lines.append(f"- #samples: {len(rows)}")
    lines.append(f"- Mean NLL (Gaussian): {mean_nll:.4f}\n")

    lines.append("## Coverage of μ ± kσ intervals\n")
    lines.append("| k | Empirical coverage | Ideal Gaussian | |Δ| |")
    lines.append("|---|---------------------|---------------|------|")

    for k, cov_emp, ideal in [
        (1.0, cov_1, ideal_1),
        (2.0, cov_2, ideal_2),
        (3.0, cov_3, ideal_3),
    ]:
        if math.isnan(cov_emp):
            lines.append(f"| {k} | NaN | {ideal:.4f} | - |")
        else:
            diff = abs(cov_emp - ideal)
            lines.append(f"| {k} | {cov_emp:.4f} | {ideal:.4f} | {diff:.4f} |")

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Wrote ECE-style summary to {OUT_MD}")


if __name__ == "__main__":
    main()
