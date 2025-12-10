from pathlib import Path
from .grid_world import GridWorld
from .astar_core import astar_search
from .heuristics import manhattan_heuristic, UncertaintyAwareHeuristic
from .multiobj_target_selector import select_target

BASE = Path(__file__).resolve().parent.parent


def make_problem_from_multiobj(start=(0, 0),
                               grid_size=(32, 32),
                               obstacles=None,
                               strategy="risk_averse"):
    """
    Χτίζει GridWorld όπου το goal έρχεται από multi-objective strategy.
    """
    width, height = grid_size

    if obstacles is None:
        obstacles = set()

    gx, gy, meta = select_target(strategy=strategy)
    goal = (gx, gy)

    problem = GridWorld(width, height, start, goal, obstacles)
    return problem, meta


def run_multiobj_astar(strategy="risk_averse"):
    print(f"[INFO] Running Multi-Objective A* with strategy: {strategy}")

    problem, meta = make_problem_from_multiobj(strategy=strategy)

    print(f"[INFO] Selected dynamic goal at: {problem.goal}")
    print(f"[INFO] Strategy metrics: {meta}")

    heuristic = UncertaintyAwareHeuristic("heuristic_net_unc.pt", k=1.0)

    stats = astar_search(problem, heuristic)

    print("---- A* RESULT ----")
    print(f"Success: {stats.success}")
    print(f"Expansions: {stats.expansions}")
    print(f"Path Cost: {stats.path_cost}")

    return stats


if __name__ == "__main__":
    run_multiobj_astar(strategy="risk_averse")
