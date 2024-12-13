import numpy as np
from adi.fullnet2 import FullNet
from cube import Cube, get_children_of
from moves import Move
import torch
from adi.fullnet2 import MSE_COSTS, SOFTMAX_COSTS
import matplotlib.pyplot as plt

ITERATIONS = 500


def generate_samples(depth: int, iterations: int):
    """Generate scrambled cube samples"""

    def get_next_move(last_move):
        while True:
            next_move = np.random.choice(list(Move))
            if next_move != last_move:
                return next_move

    class Sample:
        def __init__(self, cube, depth):
            self.cube = cube
            self.depth = depth

    for _ in range(iterations):
        c, last_move = Cube(), None
        for d in range(depth):
            last_move = move = get_next_move(last_move)
            # Create new cube and copy the state
            new_cube = Cube()
            new_cube.faces = c.faces.copy()
            new_cube.change_by(move)
            yield Sample(new_cube, d + 1)
            c = new_cube  # Update reference cube


class AutodidacticTrainer:
    def __init__(self):
        self._set_hyper()
        self._net = FullNet(
            self._body_net_sizes, self._value_net_sizes, self._policy_net_sizes
        )

    def _set_hyper(self):
        self._iteration_rounds = ITERATIONS

        # Network architecture sizes
        self._body_net_sizes = [
            14 * 6,
            512,
            128,
        ]  # Input is 84 (14*6) for one-hot encoding
        self._value_net_sizes = [self._body_net_sizes[-1], 32, 1]
        self._policy_net_sizes = [
            self._body_net_sizes[-1],
            32,
            6,
        ]  # Output is 6 for possible moves

        self._sampling_depth = 64
        self._sampling_iterations = 1024

    def train(self):
        rate = 1.0
        for iteration in range(self._iteration_rounds):
            print(f"Training iteration {iteration}/{self._iteration_rounds}")
            # if iteration % 10 == 0:
            #     print(f"Training iteration {iteration}/{self._iteration_rounds}")

            # Generate scrambled cube samples
            X = list(
                generate_samples(
                    depth=self._sampling_depth, iterations=self._sampling_iterations
                )
            )

            best_values, best_policies = [], []

            # Evaluate each sample
            for x in X:
                # Collect all children and their states
                children = list(get_children_of(x.cube))
                # Get one-hot encoding for all children at once
                child_states = np.array(
                    [child.one_hot_encode() for child in children]
                ).T
                child_states = (
                    torch.from_numpy(child_states).float().to(self._net.device)
                )

                # Get values for all children in one batch
                estimated_values = self._net.evaluate(child_states)
                values = [v.value for v in estimated_values]

                # Add rewards
                rewards = [1.0 if child.is_solved() else -1.0 for child in children]
                values = [v + r for v, r in zip(values, rewards)]

                best_values.append(np.max(values))
                best_policies.append(np.argmax(values))
            # for x in X:
            #     values = []
            #     # Get children using get_children_of function
            #     for child in get_children_of(x.cube):
            #         # Get one-hot encoding and reshape for network
            #         state = child.one_hot_encode()
            #         state = state.reshape(-1, 1)  # Make column vector
            #         state = torch.from_numpy(state).float().to(self._net.device)

            #         estimated_value = self._net.evaluate(state)[0].value
            #         reward = 1.0 if child.is_solved() else -1.0
            #         values.append(estimated_value + reward)

            #     best_values.append(np.max(values))
            #     best_policies.append(np.argmax(values))

            # Extract states and prepare for training
            cubes = [sample.cube for sample in X]
            depths = [rate / sample.depth for sample in X]
            rate *= 0.99

            # Train the network
            encoded_states = np.array([cube.one_hot_encode() for cube in cubes]).T
            encoded_states = (
                torch.from_numpy(encoded_states).float().to(self._net.device)
            )

            self._net.learn(
                X=encoded_states,
                values=best_values,
                policies=best_policies,
                weights=depths,
            )

    @property
    def net(self):
        return self._net


def train_and_save():
    trainer = AutodidacticTrainer()
    print("Starting training...")
    trainer.train()

    # Save the trained network
    import pickle

    with open(f"train_torch_{ITERATIONS}.pkl", "wb") as f:
        pickle.dump(trainer.net, f)
    print(f"Training complete! Network saved to trained_2{ITERATIONS}.pkl")

    fig, ax1 = plt.subplots()
    ax1.plot(range(len(MSE_COSTS)), MSE_COSTS, color="b")
    ax1.set_ylabel("mse cost", color="b")
    ax2 = ax1.twinx()
    ax2.plot(range(len(SOFTMAX_COSTS)), SOFTMAX_COSTS, color="g")
    ax2.set_ylabel("softmax cost", color="g")
    plt.title("Training Costs Over Time")
    plt.savefig(f"training_costs_{ITERATIONS}.png")
    plt.show()


if __name__ == "__main__":
    train_and_save()
