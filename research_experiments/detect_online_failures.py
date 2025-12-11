import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .online_multiobj_astar import online_multiobj_astar
from .multiobj_target_selector import select_target


BASE = Path(__file__).resolve().parent.parent
OUT_CSV = BASE / "phd_experiments" / "failure_cases.csv"
OUT_MD = BASE / "phd_experiments" / "failure_cases_summary.md"


# ------------------------------------------
# FAILURE DETECTORS
# ------------------------------------------

def detect_oscillation(path, threshold=3):
    """
    Αν κάποιο state εμφανίζεται >= threshold φορές → oscillation.
    """
    counts = Counter(path)
    for s, c in counts.items():
        if c >= threshold:
            return True, s, c
    return False, None, None


def detect_instability(path, eps=0.5, ratio=0.7):
    """
    Αν πάνω από ratio του path έχει μετακίνηση < eps → Zeno behavior.
    """
    if len(path) < 3:
        return False, None

    small_moves = 0
    for i in range(len(path) - 1):
        a = np.array(path[i])
        b = np.array(path[i + 1])
        d = np.linalg.norm(a - b)
        if d < eps:
            small_moves += 1

    r = small_moves / (len(path) - 1)
    return r > ratio, r


def detect_goal_chasing(goals, threshold=5):
    """
    Πολλές αλλαγές στόχου χωρίς σύγκλιση.
    """
    changes = 0
    for i in range(1, len(goals)):
        if goals[i] != goals[i - 1]:
            changes += 1

    return changes >= threshold, changes


def detect_entropy_trap(goal_metas, path_len, min_progress=0.3):
    """
    Υψηλό uncertainty αλλά μικρή πρόοδος.
    """
    if not goal_metas:
        return False, None

    mean_unc = np.mean([m["uncertainty"] for m in goal_metas])
    progress_ratio = path_len / max(len(goal_metas), 1)

    trap = (mean_unc > 40.0) and (progress_ratio < min_progress)
    return trap, mean_unc


# ------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------

def main():
    N = 30
    rows = []
    summaries = []

    print(f"[INFO] Running failure detection over {N} online runs...")

    for run_id in range(N):
        print(f"--- ONLINE RUN {run_id} ---")

        result = online_multiobj_astar(
            replanning_interval=8,
            strategy="risk_averse",
            max_steps=150,
        )

        path = result.get("path", None)
        goals = result.get("goals", None)
        goal_metas = result.get("goal_metas", None)

        if path is None:
            print("[WARN] No path returned — skipping.")
            continue

        # ---- DETECTORS ----
        osc, osc_state, osc_count = detect_oscillation(path)
        instab, instab_ratio = detect_instability(path)
        chase, num_changes = detect_goal_chasing(goals) if goals else (False, 0)
        trap, mean_unc = detect_entropy_trap(goal_metas, len(path))

        failure = osc or instab or chase or trap

        rows.append({
            "run": run_id,
            "oscillation": int(osc),
            "instability": int(instab),
            "goal_chasing": int(chase),
            "entropy_trap": int(trap),
            "failure": int(failure),
            "osc_state": osc_state,
            "osc_count": osc_count,
            "instability_ratio": instab_ratio,
            "goal_changes": num_changes,
            "mean_uncertainty": mean_unc,
        })

        if failure:
            summaries.append(rows[-1])

    # ------------------------------------------
    # SAVE CSV
    # ------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # ------------------------------------------
    # SAVE MARKDOWN SUMMARY
    # ------------------------------------------
    fails = [r for r in rows if r["failure"] == 1]

    lines = []
    lines.append("# Automatic Failure Detection for Online MO-A*\n")
    lines.append(f"Total runs: {N}")
    lines.append(f"Failures detected: {len(fails)}\n")

    lines.append("| Failure Type | Count |")
    lines.append("|--------------|-------|")
    lines.append(f"| Oscillation | {sum(r['oscillation'] for r in rows)} |")
    lines.append(f"| Instability | {sum(r['instability'] for r in rows)} |")
    lines.append(f"| Goal chasing | {sum(r['goal_chasing'] for r in rows)} |")
    lines.append(f"| Entropy trap | {sum(r['entropy_trap'] for r in rows)} |")

    lines.append("\n## Individual failure cases\n")
    lines.append("| run | osc | instab | chase | trap | goal_changes | mean_uncertainty |")
    lines.append("|-----|------|--------|--------|--------|--------------|------------------|")

    for r in fails:
        lines.append(
            f"| {r['run']} | {r['oscillation']} | {r['instability']} | "
            f"{r['goal_chasing']} | {r['entropy_trap']} | "
            f"{r['goal_changes']} | {r['mean_uncertainty']} |"
        )

    OUT_MD.write_text("\n".join(lines))

    print(f"[INFO] Saved failure cases to {OUT_CSV}")
    print(f"[INFO] Saved summary to {OUT_MD}")


if __name__ == "__main__":
    main()
