import csv
import heapq
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .grid_world import GridWorld
from .heuristics import manhattan_heuristic

# Προσπαθούμε να φέρουμε το uncertainty δίκτυο όπως στο heuristics.py
try:
    from train_heuristic_uncertainty import HeuristicNetUncertainty
    HAS_UNC_NET = True
except ImportError:
    HeuristicNetUncertainty = None
    HAS_UNC_NET = False


BASE_DIR = Path(__file__).resolve().parent.parent


def dijkstra_from_goal(problem: GridWorld) -> Dict[Tuple[int, int], float]:
    """Υπολογίζει h*(s) για όλα τα reachable states με Dijkstra από το goal."""
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


def generate_random_obstacles(width, height, density=0.1):
    obs = set()
    for x in range(width):
        for y in range(height):
            if random.random() < density:
                obs.add((x, y))
    return obs


def ensure_valid_start_goal(width, height, obstacles):
    start = (0, 0)
    goal = (width - 1, height - 1)
    obstacles.discard(start)
    obstacles.discard(goal)
    return start, goal


def make_open_grid(width, height, density=0.1):
    obs = generate_random_obstacles(width, height, density)
    start, goal = ensure_valid_start_goal(width, height, obs)
    return GridWorld(width, height, start, goal, obs)


class UncertaintyWrapper:
    """
    Τυλίγει το HeuristicNetUncertainty.
    Υποθέτουμε ότι το δίκτυο παίρνει [x, y, gx, gy] και γυρνάει (mu, log_var).
    """
    def __init__(self, model_path="heuristic_net_unc.pt", device="cpu"):
        if not HAS_UNC_NET:
            raise RuntimeError(
                "HeuristicNetUncertainty δεν είναι διαθέσιμο. "
                "Βεβαιώσου ότι υπάρχει το train_heuristic_uncertainty.py με την κλάση."
            )
        self.device = device
        self.model = HeuristicNetUncertainty()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, state, problem):
        x, y = state
        gx, gy = problem.goal
        feat = np.array([[x, y, gx, gy]], dtype=np.float32)
        inp = torch.tensor(feat, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, log_var = self.model(inp)
            mu = mu.squeeze().item()
            log_var = log_var.squeeze().item()
        sigma = float(np.sqrt(np.exp(log_var)))
        return float(mu), sigma


def run_uncertainty_error_logging(
    out_csv="phd_experiments/uncertainty_error_data.csv",
    num_instances=30,
    width=32,
    height=32,
    density=0.15,
):
    if not HAS_UNC_NET:
        print("[ERROR] HeuristicNetUncertainty δεν βρέθηκε. Δες train_heuristic_uncertainty.py.")
        return

    out_path = BASE_DIR / out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    unc = UncertaintyWrapper(model_path="heuristic_net_unc.pt", device="cpu")

    fieldnames = [
        "instance_id",
        "state_x",
        "state_y",
        "h_star",
        "mu",
        "sigma",
        "error",
        "abs_error",
        "rel_error",
        "grid_size",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        random.seed(0)
        inst_id = 0

        for _ in range(num_instances):
            inst_id += 1
            problem = make_open_grid(width, height, density=density)
            h_star_map = dijkstra_from_goal(problem)

            for (x, y), h_star in h_star_map.items():
                if (x, y) in problem.obstacles:
                    continue

                mu, sigma = unc.predict((x, y), problem)
                err = mu - h_star
                abs_err = abs(err)
                rel_err = err / (h_star + 1e-6)

                writer.writerow({
                    "instance_id": inst_id,
                    "state_x": x,
                    "state_y": y,
                    "h_star": h_star,
                    "mu": mu,
                    "sigma": sigma,
                    "error": err,
                    "abs_error": abs_err,
                    "rel_error": rel_err,
                    "grid_size": width,
                })

            print(f"[INFO] instance {inst_id}/{num_instances} processed")

    print(f"[INFO] Saved uncertainty-error log to {out_path}")


if __name__ == "__main__":
    run_uncertainty_error_logging()
