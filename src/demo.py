import pickle
import numpy as np
import time
from cube import Cube, ImmutableCube, get_children_of
from mcts.solver import Solver
from moves import Move
import matplotlib.pyplot as plt


def generate_random_cube(iterations: int = 6) -> Cube:
    """Generates a randomly scrambled 2x2 Rubik's Cube."""
    np.random.seed(1)
    cube_curr = Cube()
    scramble_moves = []
    for i in range(iterations):
        move = np.random.choice(list(Move))
        scramble_moves.append(move)
        cube_curr.change_by(move)
    return cube_curr, scramble_moves


if __name__ == "__main__":
    print("Loading neural network...")
    with open(
        "/Users/tarunvallabhaneni/Rubiks-Cube-SolverRL/src/train_torch_500.pkl", "rb"
    ) as file:
        net = pickle.load(file)
    solver = Solver(net)

    # create visualization (visualize=True)
    viz_cube = Cube(visualize=True)
    print("\nStarting from solved state...")
    time.sleep(1)

    # generate random cube and scramble moves
    cube, scramble_moves = generate_random_cube(iterations=10)

    print("\nApplying scramble...")
    for i, move in enumerate(scramble_moves, 1):
        viz_cube.change_by(move)
        print(f"Scramble move {i}: {move}")

    print("\nScrambled state reached")
    print("Is solved?", viz_cube.is_solved())
    time.sleep(1)

    # Solve the cube
    print("\nFinding solution...")
    moves = solver.solve(cube)

    if moves is not None:
        print(f"\nSolution found!")
        print(f"Number of moves: {len(moves)}")
        print("Solution sequence:", [move for move in moves])

        apply_solution = (
            input("\nSolution found! Would you like to apply it? (yes/no): ")
            .strip()
            .lower()
        )
        if apply_solution in ["yes", "y"]:
            print("\nApplying solution...")
            for i, move in enumerate(moves, 1):
                viz_cube.change_by(move)
                print(f"Solution move {i}: {move}")
                time.sleep(0.5)

            print("\nFinal state solved?", viz_cube.is_solved())

            print("\nVisualization complete. Press Enter to exit.")
            plt.ion()  # Keep interactive mode on
            plt.draw()  # Update the display

            # Wait for Enter key
            input()

            # Clean up
            plt.close("all")
        else:
            print("\nSolution not applied.")
    else:
        print("No solution found")
