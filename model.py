import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()

        hidden_dim_1 = 256
        hidden_dim_2 = 64

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_dim_1, bias=False)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2, bias=False)
        self.fc3 = nn.Linear(hidden_dim_2, action_size, bias=False)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

