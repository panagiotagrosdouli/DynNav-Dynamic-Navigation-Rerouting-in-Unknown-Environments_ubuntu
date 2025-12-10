import csv
import random
from pathlib import Path

from .grid_world import GridWorld
from .astar_core import astar_search
from .heuristics import manhattan_heuristic, LearnedHeuristic, UncertaintyAwareHeuristic
from .run_multiobj_astar import make_problem_from_multiobj

BASE = Path(__file__).resolve().parent.parent
OUT_CSV = BASE / "phd_experiments" / "multiobj_astar_stats.csv"


def make_random_problem(grid_size=(32, 32), obstacle_prob=0.15):
    w, h = grid_size
    obstacles = set()

    for x in range(w):
        for y in range(h):
            if random.random() < obstacle_prob:
                obstacles.add((x, y))

    start = (0, 0)
    obstacles.discard(start)

    # goal θα αντικατασταθεί από multi-objective στον MO case
    goal = (w - 1, h - 1)
    obstacles.discard(goal)

    return GridWorld(w, h, start, goal, obstacles), obstacles


def run_one(seed):
    random.seed(seed)

    base_problem, obstacles = make_random_problem()

    results = []

    # ---------- Classic ----------
    stats = astar_search(base_problem, manhattan_heuristic)
    results.append(("classic", stats))

    # ---------- Learned ----------
    learned_h = LearnedHeuristic("heuristic_net.pt")
    stats = astar_search(base_problem, learned_h)
    results.append(("learned", stats))

    # ---------- Uncertainty ----------
    uncertain_h = UncertaintyAwareHeuristic("heuristic_net_unc.pt", k=1.0)
    stats = astar_search(base_problem, uncertain_h)
    results.append(("uncertainty", stats))

    # ---------- Multi-Objective Driven ----------
    mo_problem, meta = make_problem_from_multiobj(
        start=base_problem.start,
        grid_size=(base_problem.width, base_problem.height),
        obstacles=obstacles,
        strategy="risk_averse",
    )

    stats = astar_search(
        mo_problem,
        UncertaintyAwareHeuristic("heuristic_net_unc.pt", k=1.0),
    )
    results.append(("multiobj_risk_averse", stats))

    return results


def main():
    N = 50   # μπορείς να το ανεβάσεις σε 100 όταν θες
    print(f"[INFO] Running {N} Monte Carlo trials...")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for i in range(N):
        run_res = run_one(seed=i)

        for method, stats in run_res:
            rows.append({
                "run": i,
                "method": method,
                "success": int(stats.success),
                "expansions": stats.expansions,
                "path_cost": stats.path_cost,
            })

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "method", "success", "expansions", "path_cost"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Wrote batch stats to {OUT_CSV}")


if __name__ == "__main__":
    main()
