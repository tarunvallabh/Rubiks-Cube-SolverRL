from typing import NamedTuple, List
import numpy as np


class NodeInfo(NamedTuple):
    is_leaf: bool
    N: List[int]  # visit counts
    W: List[float]  # max values
    L: List[float]  # virtual losses
    P: List[float]  # prior probs

    @classmethod
    def create_new(cls, probs: List[float]) -> "NodeInfo":
        size = len(probs)
        return cls(
            is_leaf=True,
            N=[0 for _ in range(size)],
            W=[0.0 for _ in range(size)],
            L=[0.0 for _ in range(size)],
            P=probs.copy(),
        )

    def get_best_action(self, exploration_factor: float) -> int:
        total_visits = np.sum(self.N)
        scores = [
            exploration_factor * self.P[i] * np.sqrt(total_visits) / (1 + self.N[i])
            + (self.W[i] - self.L[i])
            for i in range(len(self.N))
        ]
        return np.argmax(scores)

    def update_virtual_loss(self, action: int, loss_step: float):
        if not 0 <= action < len(self.L):
            raise ValueError(f"Invalid action: {action}")
        self.L[action] += loss_step

    def update_on_backup(self, action: int, loss_step: float, propagated_value: float):
        self.update_virtual_loss(action, loss_step)
        self.W[action] = max(self.W[action], propagated_value)
        self.N[action] += 1
