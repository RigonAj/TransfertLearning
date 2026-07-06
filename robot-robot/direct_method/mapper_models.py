import numpy as np
import torch
import torch.nn as nn

ARM_OBS_2DOF = 6
ARM_OBS_3DOF = 8
MAX_REACH = 3.0


class StateMapperMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden,    hidden), nn.LeakyReLU(),
            nn.Linear(hidden,    hidden), nn.LeakyReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionMapperMLP(nn.Module):
    def __init__(self, act_in_dim: int, act_out_dim: int,
                 hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(act_in_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden,    hidden), nn.LeakyReLU(),
            nn.Linear(hidden,    hidden), nn.LeakyReLU(),
            nn.Linear(hidden, act_out_dim),
        )
    def forward(self, act_in: torch.Tensor) -> torch.Tensor:
        return self.net(act_in)

