import csv
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
ERR_CSV = BASE / "phd_experiments" / "uncertainty_error_data.csv"
OUT_MD = BASE / "phd_experiments" / "uncertainty_calibration_summary.md"
PLOTS = BASE / "phd_experiments" / "uncertainty_plots"


############################################################
# Load uncertainty error data
############################################################

def load_data():
    rows = []
    with ERR_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "instance_id": int(r["instance_id"]),
                "x": int(r["state_x"]),
                "y": int(r["state_y"]),
                "h_star": float(r["h_star"]),
                "mu": float(r["mu"]),
                "sigma": float(r["sigma"]),
                "error": float(r["error"]),
                "abs_error": float(r["abs_error"]),
            })
    return rows


############################################################
# Plotting utilities
############################################################

def scatter_sigma_abs_error(sigmas, abs_errors, out_path):
    plt.figure()
    plt.scatter(sigmas, abs_errors, s=5, alpha=0.3)
    plt.xlabel("Predicted σ (uncertainty)")
    plt.ylabel("|Error| = |μ − h*|")
    plt.title("Abs Error vs Predictive Sigma")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def binned_calibration(sigmas, abs_errors, num_bins=10):
    sigmas = np.array(sigmas)
    abs_errors = np.array(abs_errors)

    mask = np.isfinite(sigmas) & np.isfinite(abs_errors)
    sigmas = sigmas[mask]
    abs_errors = abs_errors[mask]

    if len(sigmas) == 0:
        return [], [], []

    bins = np.linspace(sigmas.min(), sigmas.max(), num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_abs = []
    counts = []

    for i in range(num_bins):
        m = (sigmas >= bins[i]) & (sigmas < bins[i+1])
        if m.sum() == 0:
            mean_abs.append(float("nan"))
            counts.append(0)
        else:
            mean_abs.append(float(abs_errors[m].mean()))
            counts.append(int(m.sum()))

    return bin_centers, mean_abs, counts


def plot_binned_calibration(bin_centers, mean_abs, out_path):
    plt.figure()
    plt.plot(bin_centers, mean_abs, marker="o")
    plt.xlabel("Sigma bin center")
    plt.ylabel("Mean |Error|")
    plt.title("Binned Calibration Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


############################################################
# MAIN
############################################################

def main():
    if not ERR_CSV.exists():
        print(f"[ERROR] Missing {ERR_CSV}. Run run_uncertainty_error_logging first.")
        return

    PLOTS.mkdir(parents=True, exist_ok=True)

    data = load_data()
    sigmas = [r["sigma"] for r in data]
    abs_errors = [r["abs_error"] for r in data]

    # 1) Scatter plot
    scatter_sigma_abs_error(sigmas, abs_errors, PLOTS / "sigma_vs_abs_error.png")

    # 2) Binned calibration
    bin_centers, mean_abs, counts = binned_calibration(sigmas, abs_errors)
    if bin_centers is not None and len(bin_centers) > 0:
        plot_binned_calibration(bin_centers, mean_abs, PLOTS / "binned_calibration.png")

    # 3) Markdown summary
    valid_err = [e for e in abs_errors if math.isfinite(e)]
    mean_abs_err = sum(valid_err) / len(valid_err)

    lines = [
        "# Uncertainty Calibration Summary\n",
        f"- Total samples: {len(data)}",
        f"- Mean absolute error: {mean_abs_err:.4f}\n",
        "## Figures\n",
        "### Scatter: |error| vs sigma",
        "![](uncertainty_plots/sigma_vs_abs_error.png)\n",
        "### Binned calibration",
        "![](uncertainty_plots/binned_calibration.png)\n"
    ]

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Calibration analysis written to {OUT_MD}")


if __name__ == "__main__":
    main()
