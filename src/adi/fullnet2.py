from typing import List, NamedTuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Union


BODY_LEARNING_RATE = 0.02
POLICY_LEARNING_RATE = 0.2
VALUE_LEARNING_RATE = 0.001

POLICY_PROP_FACTOR = 0.3
VALUE_PROP_FACTOR = 0.7

MSE_COSTS = []
SOFTMAX_COSTS = []


class ValuePolicyPair(NamedTuple):
    value: float
    policy: List[float]


class BodyNet(nn.Module):
    def __init__(self, sizes: List[int]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sizes[0], sizes[1]),
            nn.ELU(alpha=0.1),
            nn.Linear(sizes[1], sizes[2]),
        )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, sizes: List[int]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sizes[0], sizes[1]), nn.ELU(alpha=0.1), nn.Linear(sizes[1], 1)
        )

    def forward(self, x):
        return self.net(x)


class PolicyNet(nn.Module):
    def __init__(self, sizes: List[int]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sizes[0], sizes[1]),
            nn.ELU(alpha=0.1),
            nn.Linear(sizes[1], sizes[2]),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.net(x))


class FullNet(nn.Module):
    def __init__(
        self,
        body_net_sizes: List[int],
        value_net_sizes: List[int],
        policy_net_sizes: List[int],
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert body_net_sizes[-1] == value_net_sizes[0] == policy_net_sizes[0]
        assert value_net_sizes[-1] == 1

        self.body_net = BodyNet(body_net_sizes)
        self.value_net = ValueNet(value_net_sizes)
        self.policy_net = PolicyNet(policy_net_sizes)

        # Initialize optimizers
        self.body_optimizer = optim.Adam(self.body_net.parameters())
        self.value_optimizer = optim.Adam(self.value_net.parameters())
        self.policy_optimizer = optim.Adam(self.policy_net.parameters())

        self.to(self.device)

    def evaluate(self, X) -> List[ValuePolicyPair]:
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X.T).float().to(self.device)
        else:  # Already a tensor
            X_tensor = X.T.float().to(self.device)

        with torch.no_grad():
            body_out = self.body_net(X_tensor)
            values = self.value_net(body_out).squeeze().cpu().tolist()
            policies = self.policy_net(body_out).cpu().tolist()

        if isinstance(values, float):  # Handle single item case
            values = [values]
            policies = [policies]

        return [ValuePolicyPair(v, p) for v, p in zip(values, policies)]

    def learn(
        self,
        X,
        values: List[float],
        policies: List[int],
        weights: List[float],
    ):
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X.T).float().to(self.device)
        else:  # Already a tensor
            X_tensor = X.T.float().to(self.device)
        values_tensor = torch.tensor(values).float().to(self.device)
        policies_tensor = torch.tensor(policies).long().to(self.device)
        weights_tensor = torch.tensor(weights).float().to(self.device)

        # Zero all gradients
        self.body_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        # Forward pass
        body_out = self.body_net(X_tensor)
        value_out = self.value_net(body_out).squeeze()
        policy_out = self.policy_net(body_out)

        # Compute losses
        value_criterion = nn.MSELoss(reduction="none")
        policy_criterion = nn.CrossEntropyLoss(reduction="none")

        value_loss = (value_criterion(value_out, values_tensor) * weights_tensor).mean()
        MSE_COSTS.append(value_loss.item())
        policy_loss = (
            policy_criterion(policy_out, policies_tensor) * weights_tensor
        ).mean()
        SOFTMAX_COSTS.append(policy_loss.item())

        # Backward pass with weighted gradients
        total_loss = value_loss * VALUE_PROP_FACTOR + policy_loss * POLICY_PROP_FACTOR
        total_loss.backward()

        # Update weights
        self.body_optimizer.step()
        self.value_optimizer.step()
        self.policy_optimizer.step()
