import random
from pathlib import Path

from .grid_world import GridWorld
from .astar_core import astar_search
from .heuristics import UncertaintyAwareHeuristic
from .multiobj_target_selector import select_target


BASE = Path(__file__).resolve().parent.parent


def make_random_problem(grid_size=(32, 32), obstacle_prob=0.15):
    w, h = grid_size
    obstacles = set()

    for x in range(w):
        for y in range(h):
            if random.random() < obstacle_prob:
                obstacles.add((x, y))

    start = (0, 0)
    obstacles.discard(start)

    dummy_goal = (w - 1, h - 1)
    obstacles.discard(dummy_goal)

    return GridWorld(w, h, start, dummy_goal, obstacles)


def online_multiobj_astar(
    replanning_interval=10,
    strategy="risk_averse",
    max_steps=200,
):
    print(f"[INFO] Online MO-A* | Strategy={strategy}, replanning_interval={replanning_interval}")

    problem = make_random_problem()
    current = problem.start

    path_traveled = [current]
    goal_history = []
    goal_meta_history = []

    total_cost = 0
    total_expansions = 0

    heuristic = UncertaintyAwareHeuristic("heuristic_net_unc.pt", k=1.0)

    for step in range(max_steps):

        # --------- Dynamic target update ----------
        if step % replanning_interval == 0:
            gx, gy, meta = select_target(strategy=strategy)
            problem.goal = (gx, gy)

            goal_history.append((gx, gy))
            goal_meta_history.append(meta)

            print(f"[STEP {step}] New dynamic goal: {problem.goal} | meta={meta}")

        # --------- Local replanning ----------
        local_problem = GridWorld(
            problem.width,
            problem.height,
            current,
            problem.goal,
            problem.obstacles,
        )

        stats = astar_search(local_problem, heuristic)
        total_expansions += stats.expansions

        if not stats.success or not stats.path or len(stats.path) < 2:
            print("[WARN] Planning failure.")
            break

        # move ONE step along the path
        next_state = stats.path[1]
        step_cost = 1.0

        total_cost += step_cost
        current = next_state
        path_traveled.append(current)

        if current == problem.goal:
            print(f"[SUCCESS] Reached dynamic goal at step {step}")
            break

    print("\n===== ONLINE MO-A* SUMMARY =====")
    print(f"Final position: {current}")
    print(f"Total steps: {len(path_traveled)-1}")
    print(f"Total expansions: {total_expansions}")
    print(f"Total path cost: {total_cost}")

    return {
        "steps": len(path_traveled) - 1,
        "expansions": total_expansions,
        "path_cost": total_cost,
        "path": path_traveled,
        "goals": goal_history,
        "goal_metas": goal_meta_history,
    }


if __name__ == "__main__":
    online_multiobj_astar(
        replanning_interval=8,
        strategy="risk_averse",
    )
