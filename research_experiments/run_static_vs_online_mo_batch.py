import csv
import random
from pathlib import Path

from .online_multiobj_astar import online_multiobj_astar
from .run_multiobj_astar import run_multiobj_astar
from .grid_world import GridWorld

BASE = Path(__file__).resolve().parent.parent
OUT_CSV = BASE / "phd_experiments" / "static_vs_online_mo_stats.csv"


def run_static_once(seed):
    random.seed(seed)
    stats = run_multiobj_astar(strategy="risk_averse")
    return {
        "expansions": stats.expansions,
        "path_cost": stats.path_cost,
        "success": int(stats.success),
    }


def run_online_once(seed):
    random.seed(seed)
    stats = online_multiobj_astar(
        replanning_interval=10,
        strategy="risk_averse",
        max_steps=200,
    )
    return {
        "expansions": stats["expansions"],
        "path_cost": stats["path_cost"],
        "success": 1,
    }


def main():
    N = 30
    print(f"[INFO] Running {N} trials for static vs online MO-A*")

    rows = []

    for i in range(N):
        print(f"--- RUN {i} ---")

        s = run_static_once(i)
        rows.append({
            "run": i,
            "mode": "static",
            **s,
        })

        o = run_online_once(i)
        rows.append({
            "run": i,
            "mode": "online",
            **o,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "mode", "success", "expansions", "path_cost"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Saved static vs online stats to {OUT_CSV}")


if __name__ == "__main__":
    main()
