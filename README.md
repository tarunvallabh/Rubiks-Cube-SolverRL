# From Scrambled to Solved: Neural-Guided Monte Carlo Tree Search for Solving 2x2x2 Rubik's Cubes

This repository contains the code for a deep reinforcement learning agent that solves the 2x2x2 Rubik's Cube using Autodidactic Iteration (ADI) and Monte Carlo Tree Search (MCTS). This project is inspired by and builds upon the work of McAleer et al. (DeepCube).

## File Structure

The codebase is organized as follows:

* **`/adi`**: Contains modules related to the neural network and Autodidactic Iteration.
  * `fullnet2.py`: Defines the neural network architecture using PyTorch.
* **`/mcts`**: Contains modules related to Monte Carlo Tree Search.
  * `solver.py`: Implements the MCTS solver, which uses the trained neural network to guide the search.
  * `node_info.py`: Defines the `NodeInfo` class, which stores information about each node in the MCTS tree.
  * `bfser.py`: Implements the `BFSer` class, which performs a breadth-first search to find the shortest solution path.
* **`cube.py`**: Defines the `Cube` and `ImmutableCube` classes, representing the 2x2x2 Rubik's Cube state and move logic.
* **`moves.py`**: Defines the `Move` enum, representing the possible moves on the cube.
* **`train.py`**: Contains the code for training the neural network using Autodidactic Iteration.
* **`test.py`**: Contains code for testing the trained agent on a set of scrambled cubes.
* **`experiments.py`**: Contains code for running experiments to evaluate the agent's performance across different scramble depths and generate statistics.
* **`demo.py`**: Contains a script to run an interactive demonstration of the solver, allowing users to scramble a cube and watch the agent solve it.
* **`train_torch_500.pkl`**: Saved file for the trained neural network.

## Requirements

* Python 3.7+
* NumPy
* PyTorch
* matplotlib
* mpl_toolkits (for 3D visualization)
* pandas

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/tarunvallabh/Rubiks-Cube-SolverRL.git
   cd Rubiks-Cube-SolverRL
   ```

2. Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Demo

To run an interactive demonstration of the solver, execute the following command:
```bash
python demo.py
```

This will launch a visual representation of a 2x2x2 Rubik's cube. The script will then:
1. Scramble the cube randomly
2. Use the trained neural network and MCTS to find a solution
3. Display the solution moves

You will be prompted if you want to apply the solution to the visualized cube.

### Training

To train the neural network from scratch, run:
```bash
python train.py
```

This will train the network using the parameters defined in train.py and save the trained model to a .pkl file.

**Note:** Training can take a significant amount of time.


### Experiments

To run experiments and generate performance statistics, run:
```bash
python experiments.py
```

This will generate plots of success rate, solution length, solve time, and MCTS tree size for different scramble depths. The plots will be saved in the extra directory. The raw data will be saved to experiment_data.csv.