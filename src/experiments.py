import pickle
import time
from typing import Tuple, List

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend

import numpy as np
import pandas as pd

from cube import Cube
from moves import Move
from mcts.solver import Solver
import os


def generate_random_cube(scramble_length: int = 10) -> Cube:
    """Generates a randomly scrambled 2x2 Rubik's Cube."""
    cube = Cube()
    for _ in range(scramble_length):
        move = np.random.choice(list(Move))
        cube.change_by(move)
    return cube


def measure_effectiveness(
    net_path: str, scramble_range: int, num_cubes: int, time_limit: int
) -> Tuple[List[int], List[float], List[int], float, List[float], List[int]]:
    """Measures the effectiveness of a trained network on solving random cubes."""

    with open(net_path, "rb") as f:
        net = pickle.load(f)

    solver = Solver(net)

    tree_sizes, solve_times, solution_lengths = [], [], []
    success_count = 0
    # Initialize lists to store data for each cube
    all_tree_sizes = []
    all_solve_times = []
    all_solution_lengths = []

    for i in range(num_cubes):
        cube = generate_random_cube(scramble_length=scramble_range)
        start_time = time.time()
        solution = solver.solve(cube, timeout=time_limit)
        end_time = time.time()

        if solution is not None:
            success_count += 1
            tree_size = len(solver._tree)
            solve_time = end_time - start_time
            solution_length = len(solution)

            tree_sizes.append(tree_size)
            solve_times.append(solve_time)
            solution_lengths.append(solution_length)

            # Store data for this cube
            all_tree_sizes.append(tree_size)
            all_solve_times.append(solve_time)
            all_solution_lengths.append(solution_length)
        else:
            # If no solution is found, log a DNF
            all_tree_sizes.append(np.nan)
            all_solve_times.append(np.nan)
            all_solution_lengths.append(np.nan)

        # Progress indicator
        if (i + 1) % 10 == 0:  # Print progress every 10 cubes
            print(f"Processed {i + 1} out of {num_cubes} cubes")

    success_rate = (success_count / num_cubes) * 100
    return (
        tree_sizes,
        solve_times,
        solution_lengths,
        success_rate,
        all_tree_sizes,
        all_solve_times,
        all_solution_lengths,
    )


def plot_stats(
    num_cubes: int,
    scramble_range: int,
    solve_times: List[float],
    tree_sizes: List[int],
    solution_lengths: List[int],
    success_rate: float,
):
    """Plots statistics about the solver's performance."""
    fig, axs = plt.subplots(3, figsize=(8, 10))
    plt.suptitle(
        f"Solver Effectiveness on {num_cubes} cubes (up to {scramble_range} moves)\n"
        f"Success Rate: {success_rate:.2f}%"
    )

    # Time Plot
    axs[0].hist(solve_times, bins=20, color="skyblue", edgecolor="black")
    axs[0].set_title("Time Taken to Solve")
    axs[0].set_xlabel("Time (seconds)")
    axs[0].set_ylabel("Number of Cubes")
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # Tree Size Plot
    axs[1].hist(tree_sizes, bins=20, color="lightgreen", edgecolor="black")
    axs[1].set_title("MCTS Tree Size")
    axs[1].set_xlabel("Number of Nodes in Tree")
    axs[1].set_ylabel("Number of Cubes")
    axs[1].grid(True, linestyle="--", alpha=0.6)

    # Solution Length Plot
    axs[2].hist(
        solution_lengths,
        bins=range(min(solution_lengths), max(solution_lengths) + 2),
        color="salmon",
        edgecolor="black",
        align="left",
    )
    axs[2].set_title("Solution Length")
    axs[2].set_xlabel("Number of Moves")
    axs[2].set_ylabel("Number of Cubes")
    axs[2].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout(pad=3.0)
    plt.savefig(
        f"/Users/tarunvallabhaneni/Rubiks-Cube-SolverRL/src/extra/experiment_{scramble_range}.png"
    )
    print(f"Results saved to extra/experiment_{scramble_range}.png")


if __name__ == "__main__":
    # Experiment parameters
    NET_PATH = "/Users/tarunvallabhaneni/Rubiks-Cube-SolverRL/src/train_torch_500.pkl"
    NUM_CUBES = 50
    TIME_LIMIT = 180

    scramble_ranges = [5, 10, 15, 20]  # Different scramble lengths to test
    num_cubes = [50, 50, 25, 10]  # Number of cubes to test for each scramble length
    time_limit = [100, 300, 600, 600]
    # print("Directory exists:", os.path.exists("results"))

    all_data = []  # List to store all experiment data

    for i, scramble_range in enumerate(scramble_ranges):
        print(f"\nRunning experiment with scramble range: {scramble_range}")
        (
            tree_sizes,
            solve_times,
            solution_lengths,
            success_rate,
            all_tree_sizes,
            all_solve_times,
            all_solution_lengths,
        ) = measure_effectiveness(NET_PATH, scramble_range, num_cubes[i], time_limit[i])
        plot_stats(
            num_cubes[i],
            scramble_range,
            solve_times,
            tree_sizes,
            solution_lengths,
            success_rate,
        )

        # Calculate and print averages
        avg_tree_size = np.mean(tree_sizes) if tree_sizes else 0
        avg_solve_time = np.mean(solve_times) if solve_times else 0
        avg_solution_length = np.mean(solution_lengths) if solution_lengths else 0

        print(f"Average Tree Size: {avg_tree_size:.2f}")
        print(f"Average Solve Time: {avg_solve_time:.2f} seconds")
        print(f"Average Solution Length: {avg_solution_length:.2f} moves")

        # Store data for CSV
        data = {
            "scramble_range": [scramble_range] * num_cubes[i],
            "cube_number": list(range(1, num_cubes[i] + 1)),
            "success": [
                1 if x is not None and not np.isnan(x) else 0
                for x in all_solution_lengths
            ],
            "solve_time": all_solve_times,
            "tree_size": all_tree_sizes,
            "solution_length": all_solution_lengths,
        }

        all_data.extend(list(zip(*data.values())))

    # Write data to CSV
    df = pd.DataFrame(
        all_data,
        columns=[
            "scramble_range",
            "cube_number",
            "success",
            "solve_time",
            "tree_size",
            "solution_length",
        ],
    )
    df.to_csv(
        "/Users/tarunvallabhaneni/Rubiks-Cube-SolverRL/src/extra/experiment_data.csv",
        index=False,
    )
    print("Experiment data saved to extra/experiment_data.csv")

    print("Experiments completed.")
