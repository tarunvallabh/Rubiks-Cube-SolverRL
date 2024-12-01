import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from cube import Cube
from encoding import encode_full, decode_full
from network import CubeNet, CubeLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random


class CubeADITrainer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Initialize network and move to device
        self.device = device
        self.net = CubeNet().to(device)

        # Optimizer with paper's learning rates
        self.optimizer = optim.Adam(
            [
                {"params": self.net.body.parameters(), "lr": 0.02},
                {"params": self.net.value_head.parameters(), "lr": 0.001},
                {"params": self.net.policy_head.parameters(), "lr": 0.2},
            ]
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=True,
        )

        # Loss function with paper's weightings
        self.criterion = CubeLoss(value_weight=0.7, policy_weight=0.3)

        # Training parameters
        self.max_depth = 10  # Adjusted for 2x2 from paper's 64
        self.iterations = 500  # Paper's iteration rounds
        self.samples_per_iter = 32  # Paper's sampling iterations

        # Available moves for 2x2
        self.moves = ["R", "R'", "U", "U'", "F", "F'"]

    def generate_samples(self, num_samples):
        """Generate scrambled cube samples following paper's method"""
        samples = []

        for _ in range(num_samples):
            cube = Cube(visualize=False)
            depth = random.randint(1, self.max_depth)
            last_move = None
            # Apply random moves avoiding back-to-back same face moves
            for d in range(depth):
                while True:
                    move = random.choice(self.moves)
                    # Don't do same face moves back to back
                    if last_move is None or move[0] != last_move[0]:
                        break
                cube.execute_move(move)
                last_move = move
            samples.append((cube, depth))
        return samples

    def evaluate_position(self, cube):
        """Try all moves and find best one based on network evaluation"""
        best_value = float("-inf")
        best_move_idx = None
        values = []
        # Try each possible move
        for i, move in enumerate(self.moves):
            next_cube = Cube()
            next_cube.faces = cube.faces.copy()
            next_cube.execute_move(move)
            # Encode position for network
            state = (
                torch.tensor(encode_full(next_cube.faces)).unsqueeze(0).to(self.device)
            )
            # Get network's evaluation
            with torch.no_grad():
                value, _ = self.net(state)
                value = value.item()
            # Add immediate reward (+1 for solved, -1 otherwise)
            reward = 1.0 if next_cube.is_solved() else -1.0
            total_value = value + reward
            values.append(total_value)

            if total_value > best_value:
                best_value = total_value
                best_move_idx = i

        return best_move_idx, best_value, values

    def train_iteration(self):
        """Single training iteration"""
        self.net.train()

        # Generate samples
        samples = self.generate_samples(self.samples_per_iter)

        # Collect states and targets
        states = []
        target_values = []
        target_policies = []
        weights = []

        # Evaluate each position
        for cube, depth in samples:
            best_move_idx, best_value, _ = self.evaluate_position(cube)

            # Store training data
            states.append(encode_full(cube.faces))
            target_values.append(best_value)
            target_policies.append(best_move_idx)
            weights.append(1.0 / depth)  # Weight by inverse depth

        # Convert to tensors
        states = torch.stack([encode_full(cube.faces) for cube, depth in samples]).to(
            self.device
        )
        target_values = torch.tensor(target_values).unsqueeze(1).to(self.device)
        target_policies = torch.tensor(target_policies).to(self.device)
        weights = torch.tensor(weights).to(self.device)

        # Forward pass
        value_pred, policy_logits = self.net(states)

        # Calculate loss
        loss, value_loss, policy_loss = self.criterion(
            value_pred, target_values, policy_logits, target_policies
        )

        # Apply sample weights
        loss = loss * weights.mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), value_loss.item(), policy_loss.item()

    def evaluate_solving(self, num_cubes=100, max_moves=50):
        """Evaluate network's solving capability"""
        self.net.eval()
        solved = 0
        total_moves = 0

        for _ in range(num_cubes):
            # Generate scrambled cube
            cube, _ = self.generate_samples(1)[0]
            moves_taken = 0

            # Try to solve
            while moves_taken < max_moves and not cube.is_solved():
                best_move_idx, _, _ = self.evaluate_position(cube)
                cube.execute_move(self.moves[best_move_idx])
                moves_taken += 1

            if cube.is_solved():
                solved += 1
                total_moves += moves_taken

        avg_moves = total_moves / solved if solved > 0 else float("inf")
        return solved / num_cubes, avg_moves

    def train(self):
        """Full training loop"""
        best_solve_rate = 0
        pbar = tqdm(range(self.iterations))

        for iteration in pbar:
            # Training iteration
            loss, value_loss, policy_loss = self.train_iteration()

            # Scheduler step
            self.scheduler.step(loss)

            # Evaluate every 10 iterations
            if iteration % 10 == 0:
                solve_rate, avg_moves = self.evaluate_solving()

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "solve_rate": f"{solve_rate:.2%}",
                        "avg_moves": f"{avg_moves:.1f}",
                    }
                )

                # Save best model
                if solve_rate > best_solve_rate:
                    best_solve_rate = solve_rate
                    self.save_model("best_cube_solver.pt")

    def save_model(self, path):
        """Save model checkpoint"""
        torch.save(
            {
                "net_state": self.net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint["net_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])


if __name__ == "__main__":
    trainer = CubeADITrainer()
    trainer.train()
