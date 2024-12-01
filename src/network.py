# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CubeNet(nn.Module):
    def __init__(self):
        super(CubeNet, self).__init__()

        # Input dimensions: 144 (24 stickers × 6 colors)
        # Shared layers (body network)
        self.body = nn.Sequential(
            nn.Linear(144, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )

        # Value head - predicts how close we are to solved state
        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # Output between -1 and 1
        )

        # Policy head - predicts move probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 6 possible moves: R, R', U, U', F, F'
            # Note: We don't include Softmax here as it's included in CrossEntropyLoss
        )

    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Tensor of shape (batch_size, 144) containing encoded cube states
        Returns:
            value: Tensor of shape (batch_size, 1) containing state values
            policy: Tensor of shape (batch_size, 6) containing move probabilities
        """
        # Ensure input is float tensor
        x = x.float()

        # Pass through shared layers
        features = self.body(x)

        # Get value and policy predictions
        value = self.value_head(features)
        policy_logits = self.policy_head(features)

        return value, policy_logits

    def get_move_probs(self, x):
        """
        Get move probabilities (useful for inference)
        """
        _, policy_logits = self.forward(x)
        return F.softmax(policy_logits, dim=1)


# Training utilities
class CubeLoss:
    def __init__(self, value_weight=1.0, policy_weight=1.0):
        self.value_weight = value_weight
        self.policy_weight = policy_weight

        # MSE for value head
        self.value_criterion = nn.MSELoss()
        # Cross entropy for policy head
        self.policy_criterion = nn.CrossEntropyLoss()

    def __call__(self, value_pred, value_target, policy_logits, policy_target):
        """
        Calculate combined loss
        Args:
            value_pred: Predicted values (batch_size, 1)
            value_target: Target values (batch_size, 1)
            policy_logits: Policy logits (batch_size, 6)
            policy_target: Target moves (batch_size,) as indices
        Returns:
            total_loss: Combined weighted loss
            value_loss: Value prediction loss component
            policy_loss: Policy prediction loss component
        """
        value_loss = self.value_criterion(value_pred, value_target)
        policy_loss = self.policy_criterion(policy_logits, policy_target)

        # Combine losses
        total_loss = self.value_weight * value_loss + self.policy_weight * policy_loss

        return total_loss, value_loss, policy_loss
