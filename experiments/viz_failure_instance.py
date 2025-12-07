import heapq
import random
from pathlib import Path
from typing import Dict, Tuple, Set

import numpy as np
import torch
import matplotlib.pyplot as plt

from .grid_world import GridWorld
from .astar_core import astar_search
from .heuristics import manhattan_heuristic
from learned_heuristic import HeuristicNet


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "phd_experiments" / "failure_viz"


########################################
# Dijkstra for h*(s)
########################################

def dijkstra_from_goal(problem: GridWorld) -> Dict[Tuple[int, int], float]:
    goal = problem.goal
    dist = {goal: 0.0}
    pq = [(0.0, goal)]
    while pq:
        d, s = heapq.heappop(pq)
        if d > dist[s]:
            continue
        for s_next, cost in problem.successors(s):
            alt = d + cost
            if s_next not in dist or alt < dist[s_next]:
                dist[s_next] = alt
                heapq.heappush(pq, (alt, s_next))
    return dist


########################################
# Random grids
########################################

def generate_random_obstacles(width: int, height: int, density: float) -> Set[Tuple[int, int]]:
    obs = set()
    for x in range(width):
        for y in range(height):
            if random.random() < density:
                obs.add((x, y))
    return obs


def ensure_valid_start_goal(width: int, height: int, obstacles: Set[Tuple[int, int]]):
    start = (0, 0)
    goal = (width - 1, height - 1)
    obstacles.discard(start)
    obstacles.discard(goal)
    return start, goal


def make_open_grid(width: int, height: int, density: float) -> GridWorld:
    obstacles = generate_random_obstacles(width, height, density)
    start, goal = ensure_valid_start_goal(width, height, obstacles)
    return GridWorld(width, height, start, goal, obstacles)


########################################
# Learned heuristic wrapper
########################################

class LearnedH:
    def __init__(self, model_path="heuristic_net.pt", stats_path="planner_dataset_norm_stats.npz", hidden=64):
        data = np.load(stats_path)
        self.mean = data["mean"]
        self.std = data["std"]

        self.model = HeuristicNet(in_dim=4, hidden=hidden)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def __call__(self, s, problem: GridWorld) -> float:
        x, y = s
        gx, gy = problem.goal
        feat = np.array([x, y, gx, gy], dtype=np.float32)
        feat = (feat - self.mean) / self.std
        inp = torch.tensor([feat], dtype=torch.float32)
        with torch.no_grad():
            h = self.model(inp).squeeze().item()
        return max(0.0, float(h))


########################################
# Failure criteria
########################################

def compute_error_stats(problem: GridWorld, h_star_map, learned_h: LearnedH):
    errors = []
    abs_errors = []
    over_count = 0
    total = 0

    for (x, y), h_star in h_star_map.items():
        s = (x, y)
        if s in problem.obstacles:
            continue
        h_pred = learned_h(s, problem)
        err = h_pred - h_star
        errors.append(err)
        abs_errors.append(abs(err))
        total += 1
        if err > 0:
            over_count += 1

    if total == 0:
        return None

    mean_abs = float(np.mean(abs_errors))
    over_rate = over_count / total
    return {
        "mean_abs_error": mean_abs,
        "over_rate": over_rate,
        "num_states": total,
    }


def find_failure_instance(
    max_tries: int = 100,
    width: int = 32,
    height: int = 32,
    density: float = 0.15,
    th_abs_error: float = 5.0,
    th_over_rate: float = 0.7,
    th_subopt: float = 1.05,
):
    """
    Γεννά τυχαία grids και ψάχνει ένα instance όπου:
      - mean_abs_error > th_abs_error
        ή over_rate > th_over_rate
        ή C_learn / C* > th_subopt
    Επιστρέφει (problem, h_star_map, error_stats, astar_stats_dict)
    """

    learned_h = LearnedH()

    for attempt in range(1, max_tries + 1):
        problem = make_open_grid(width, height, density)
        h_star_map = dijkstra_from_goal(problem)
        err_stats = compute_error_stats(problem, h_star_map, learned_h)
        if err_stats is None:
            continue

        # A* classic
        stats_classic = astar_search(problem, manhattan_heuristic)
        # A* learned
        stats_learned = astar_search(problem, lambda s, p: learned_h(s, p))

        if not (stats_classic.success and stats_learned.success):
            continue

        c_star = stats_classic.path_cost
        c_learn = stats_learned.path_cost
        subopt = c_learn / (c_star + 1e-6)

        fail_abs = err_stats["mean_abs_error"] > th_abs_error
        fail_over = err_stats["over_rate"] > th_over_rate
        fail_sub = subopt > th_subopt

        print(
            f"[TRY {attempt}] mean_abs={err_stats['mean_abs_error']:.2f}, "
            f"over_rate={err_stats['over_rate']:.2f}, subopt={subopt:.3f}"
        )

        if fail_abs or fail_over or fail_sub:
            print("[INFO] Found failure-like instance!")
            astar_stats = {
                "classic": stats_classic,
                "learned": stats_learned,
                "subopt": subopt,
            }
            return problem, h_star_map, err_stats, astar_stats

    print("[WARN] No failure instance found within max_tries")
    return None, None, None, None


########################################
# Visualization
########################################

def plot_heatmaps(problem: GridWorld, h_star_map, learned_h: LearnedH):
    width, height = problem.width, problem.height
    H_star = np.full((width, height), np.nan, dtype=np.float32)
    H_pred = np.full((width, height), np.nan, dtype=np.float32)
    H_err = np.full((width, height), np.nan, dtype=np.float32)

    for x in range(width):
        for y in range(height):
            s = (x, y)
            if s in problem.obstacles:
                continue
            h_star = h_star_map.get(s, np.nan)
            h_pred = learned_h(s, problem)
            H_star[x, y] = h_star
            H_pred[x, y] = h_pred
            if not np.isnan(h_star):
                H_err[x, y] = h_pred - h_star

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axs[0].imshow(H_star.T, origin="lower", cmap="viridis")
    axs[0].set_title("h*(s)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(H_pred.T, origin="lower", cmap="viridis")
    axs[1].set_title("h_pred(s)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(H_err.T, origin="lower", cmap="bwr")
    axs[2].set_title("error = h_pred − h*")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "failure_instance_heatmaps.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[INFO] Saved heatmaps to {out_file}")


def plot_paths(problem: GridWorld, stats_classic, stats_learned, subopt: float):
    width, height = problem.width, problem.height

    grid = np.zeros((width, height), dtype=np.int32)
    for x, y in problem.obstacles:
        grid[x, y] = 1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid.T, origin="lower", cmap="gray_r")

    # paths (υποθέτουμε ότι stats.path είναι λίστα (x,y))
    path_c = getattr(stats_classic, "path", None)
    path_l = getattr(stats_learned, "path", None)

    if path_c:
        xs = [p[0] for p in path_c]
        ys = [p[1] for p in path_c]
        ax.plot(xs, ys, "-", label="classic A*", linewidth=2)
    if path_l:
        xs = [p[0] for p in path_l]
        ys = [p[1] for p in path_l]
        ax.plot(xs, ys, "--", label="learned A*", linewidth=2)

    # start & goal
    sx, sy = problem.start
    gx, gy = problem.goal
    ax.scatter([sx], [sy], marker="o", s=60, label="start")
    ax.scatter([gx], [gy], marker="*", s=90, label="goal")

    ax.set_title(f"Failure instance paths (subopt={subopt:.3f})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "failure_instance_paths.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"[INFO] Saved paths plot to {out_file}")


########################################
# MAIN
########################################

def main():
    random.seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    problem, h_star_map, err_stats, astar_stats = find_failure_instance()
    if problem is None:
        print("[ERROR] Could not find failure instance.")
        return

    print("[INFO] Failure instance stats:")
    print(f"  mean_abs_error = {err_stats['mean_abs_error']:.2f}")
    print(f"  over_rate      = {err_stats['over_rate']:.2f}")
    print(f"  subopt         = {astar_stats['subopt']:.3f}")
    print(f"  classic_exp    = {astar_stats['classic'].expansions}")
    print(f"  learned_exp    = {astar_stats['learned'].expansions}")

    learned_h = LearnedH()
    plot_heatmaps(problem, h_star_map, learned_h)
    plot_paths(problem, astar_stats["classic"], astar_stats["learned"], astar_stats["subopt"])


if __name__ == "__main__":
    main()
