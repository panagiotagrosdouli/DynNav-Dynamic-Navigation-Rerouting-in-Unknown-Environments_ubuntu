from .grid_world import GridWorld
from .astar_core import astar_search
from .heuristics import manhattan_heuristic, LearnedHeuristic, UncertaintyAwareHeuristic
from .run_multiobj_astar import make_problem_from_multiobj


def run():
    strategies = [
        "min_uncertainty",
        "max_coverage_per_meter",
        "risk_averse",
        "risk_seeking",
    ]

    print("\n===== BASELINE: CLASSIC A* =====")
    base_problem, _ = make_problem_from_multiobj(strategy="risk_averse")
    classic = astar_search(base_problem, manhattan_heuristic)
    print(f"Expansions: {classic.expansions}, Cost: {classic.path_cost}")

    print("\n===== LEARNED A* =====")
    learned = astar_search(base_problem, LearnedHeuristic("heuristic_net.pt"))
    print(f"Expansions: {learned.expansions}, Cost: {learned.path_cost}")

    print("\n===== UNCERTAINTY A* =====")
    unc = astar_search(
        base_problem,
        UncertaintyAwareHeuristic("heuristic_net_unc.pt", k=1.0),
    )
    print(f"Expansions: {unc.expansions}, Cost: {unc.path_cost}")

    print("\n===== MULTI-OBJECTIVE DRIVEN A* =====")
    for s in strategies:
        prob, meta = make_problem_from_multiobj(strategy=s)
        stats = astar_search(
            prob,
            UncertaintyAwareHeuristic("heuristic_net_unc.pt", k=1.0),
        )

        print(f"\nStrategy: {s}")
        print(f"Goal: {prob.goal}")
        print(f"Meta: {meta}")
        print(f"Expansions: {stats.expansions}, Cost: {stats.path_cost}")


if __name__ == "__main__":
    run()
