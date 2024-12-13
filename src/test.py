import numpy as np
from cube import Cube, Move
from solver.mcts import Solver
import pickle
import time


def scramble_cube(cube: Cube, num_moves: int) -> list[Move]:
    """Scramble the cube with random moves and return the sequence"""
    moves = []
    move_list = list(Move)  # Convert enum to list once
    for _ in range(num_moves):
        # Select from actual Move enum objects
        move = move_list[np.random.randint(len(move_list))]
        moves.append(move)
        cube.change_by(move)
    return moves


def test_solve(net, scramble_depth: int, timeout: int = 60, visualize: bool = True):
    """Test MCTS solver on a scrambled cube"""
    # Create and scramble cube
    cube = Cube(visualize=visualize)
    scramble_sequence = scramble_cube(cube, scramble_depth)

    print(f"\nTesting {scramble_depth} move scramble:")
    print("Scramble sequence:", [move.name for move in scramble_sequence])

    if visualize:
        print("Initial state:")
        cube.print_raw_arrays()

    # Try to solve
    solver = Solver(net)
    start_time = time.time()
    solution = solver.solve(cube, timeout=timeout)
    solve_time = time.time() - start_time

    if solution is not None:
        print(f"Found solution in {solve_time:.2f} seconds!")
        print("Solution length:", len(solution))
        print("Solution sequence:", [move.name for move in solution])

        # Verify solution
        for move in solution:
            cube.change_by(move)

        if cube.is_solved():
            print("✓ Solution verified - cube is solved!")
            if visualize:
                print("Final state:")
                cube.print_raw_arrays()
        else:
            print("✗ Error - cube is not solved!")
    else:
        print(f"No solution found within {timeout} seconds")


def main():
    # Load the trained network
    print("Loading trained network...")
    with open("trained_500.pkl", "rb") as f:
        net = pickle.load(f)
    print("Network loaded!")

    # Test with different scramble depths
    for scramble_depth in [1, 3, 5, 7]:
        test_solve(net, scramble_depth, timeout=60, visualize=True)

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
