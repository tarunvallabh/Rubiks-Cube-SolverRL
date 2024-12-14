from typing import List, NamedTuple
from numpy import sqrt, sum, argmax


class NodeInfo(NamedTuple):
    is_leaf: bool
    N: List[int]  # visit counts
    W: List[float]  # max values
    L: List[float]  # virtual losses
    P: List[float]  # prior probs

    @classmethod
    def create_new(cls, probability_vector):
        action_space = len(probability_vector)
        zeros_int = [0] * action_space
        zeros_float = [0.0] * action_space
        return cls(
            is_leaf=True,
            N=zeros_int,
            W=zeros_float.copy(),
            L=zeros_float.copy(),
            P=probability_vector.copy(),
        )

    def get_best_action(self, exploration_factor):
        total_node_visits = sum(self.N)
        exploration_term = [
            exploration_factor * prior * sqrt(total_node_visits) / (1 + visits)
            for prior, visits in zip(self.P, self.N)
        ]
        exploitation_term = [max_val - vloss for max_val, vloss in zip(self.W, self.L)]
        combined_scores = [
            explore + exploit
            for explore, exploit in zip(exploration_term, exploitation_term)
        ]
        return argmax(combined_scores)

    def update_virtual_loss(self, chosen_action, loss_increment):
        if not 0 <= chosen_action < len(self.L):
            raise IndexError(f"Action {chosen_action} out of bounds")
        self.L[chosen_action] += loss_increment

    def update_on_backup(self, chosen_action, loss_increment, new_value):
        self.update_virtual_loss(chosen_action, loss_increment)
        self.W[chosen_action] = max(self.W[chosen_action], new_value)
        self.N[chosen_action] += 1
