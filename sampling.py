import numpy as np
import copy
import pickle
import time
from tqdm import tqdm
from cube import Cube

MOVES = {
    1: "R",
    -1: "R'",
    2: "U",
    -2: "U'",
    3: "F",
    -3: "F'",
}


def get_next_move(last_move=None):
    while True:
        move = np.random.choice(list(MOVES.keys()))
        if last_move is None or abs(move) != abs(last_move):
            return move


def generate_samples(depth: int = 30, iterations: int = 32, visual_delay=0.5):
    """Generate full training dataset"""
    samples = []
    total = depth * iterations

    print(f"Generating {total} samples...")
    pbar = tqdm(total=total)

    for each in range(iterations):
        # print(f"\nIteration {iter + 1}/{iterations}")
        cube = Cube(visualize=False)
        # if each == 0:  # Only plot the first cube
        #     cube.plot_3d_cube()
        #     time.sleep(visual_delay)
        last_move = None

        for d in range(depth):
            move = get_next_move(last_move)
            last_move = move
            # print(f"Applying move: {MOVES[move]}")

            cube.execute_move(MOVES[move], visualize=False)
            samples.append((copy.deepcopy(cube.faces), d + 1))
            pbar.update(1)
    pbar.close()
    return samples


if __name__ == "__main__":
    # First generate small test dataset
    print("Generating test dataset...")
    test_depth = 30
    test_iterations = 32
    test_samples = generate_samples(test_depth, test_iterations)

    test_filename = f"cube_samples_test_d{test_depth}_i{test_iterations}.pkl"
    with open(test_filename, "wb") as f:
        pickle.dump(test_samples, f)
    print(f"Saved test dataset ({len(test_samples)} samples) to {test_filename}")

    # If test looks good, generate full dataset
    input("\nPress Enter to generate full dataset...")

    print("Generating full dataset...")
    full_depth = 64
    full_iterations = 32 * 500  # 500 rounds
    full_samples = generate_samples(full_depth, full_iterations)

    full_filename = f"cube_samples_full_d{full_depth}_i{full_iterations}.pkl"
    with open(full_filename, "wb") as f:
        pickle.dump(full_samples, f)
    print(f"Saved full dataset ({len(full_samples)} samples) to {full_filename}")
