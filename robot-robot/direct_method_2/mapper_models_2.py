import torch
import torch.nn as nn


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
    """
    Action mapper conditioned on 2DoF joint configuration.
    
    Input:  [dtheta_2dof (2), q_2dof (2)] = 4D
    Output: dtheta_3dof (3)
    """
    def __init__(self, act_in_dim: int, act_out_dim: int,
                 q_dim: int = 2,  # 2DoF joint positions
                 hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.act_in_dim = act_in_dim
        self.act_out_dim = act_out_dim
        self.q_dim = q_dim
        total_in = act_in_dim + q_dim
        
        self.net = nn.Sequential(
            nn.Linear(total_in, hidden), nn.LeakyReLU(),
            nn.Linear(hidden,    hidden), nn.LeakyReLU(),
            nn.Linear(hidden,    hidden), nn.LeakyReLU(),
            nn.Linear(hidden, act_out_dim),
        )

    def forward(self, act_in: torch.Tensor, q_2dof: torch.Tensor) -> torch.Tensor:
        """
        Args:
            act_in: [B, 2] - 2DoF joint velocities (normalized)
            q_2dof: [B, 2] - 2DoF joint positions (normalized)
        Returns:
            [B, 3] - 3DoF joint velocities (normalized)
        """
        x = torch.cat([act_in, q_2dof], dim=-1)
        return self.net(x)