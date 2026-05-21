"""
UNN Bases — Fixed Cartesian latent space and learned action mappers.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class CartesianStateEncoder:
    """
    Encode raw observation (arm + task) into a fixed Cartesian latent state.
    For PushBall: [eff_x, eff_y, ball_x, ball_y, tgt_x, tgt_y] normalized by max_reach.
    """
    def __init__(self, max_reach: float = 3.0):
        self.max_reach = max_reach

    def encode(self, obs: np.ndarray, arm_obs_size: int) -> np.ndarray:
        """
        obs: full observation (arm_obs + task_obs)
        arm_obs_size: 6 for 2DoF, 8 for 3DoF
        Returns: latent state (6D) in [-1, 1]
        """
        # End‑effector position is the last two entries of arm_obs
        if arm_obs_size == 6:      # 2DoF: arm_obs = [θ1/π, θ2/π, dθ1/ω, dθ2/ω, eff_x/r, eff_y/r]
            eff_x = obs[4] * self.max_reach
            eff_y = obs[5] * self.max_reach
        else:                       # 3DoF: arm_obs = [θ1/π, θ2/π, θ3/π, dθ1/ω, dθ2/ω, dθ3/ω, eff_x/r, eff_y/r]
            eff_x = obs[6] * self.max_reach
            eff_y = obs[7] * self.max_reach

        # Task part: ball and target positions (already normalized by max_reach)
        ball_x = obs[arm_obs_size] * self.max_reach
        ball_y = obs[arm_obs_size + 1] * self.max_reach
        tgt_x  = obs[arm_obs_size + 2] * self.max_reach
        tgt_y  = obs[arm_obs_size + 3] * self.max_reach

        # Normalize latent state to [-1, 1]
        norm = self.max_reach
        latent = np.array([
            eff_x / norm, eff_y / norm,
            ball_x / norm, ball_y / norm,
            tgt_x / norm, tgt_y / norm
        ], dtype=np.float32)
        return np.clip(latent, -1.0, 1.0)


class ActionMapper(nn.Module):
    """
    Neural network mapping (joint_angles, desired_ee_displacement) → joint_action (Δθ).
    Uses current joint angles (normalized by π) to provide kinematic context.
    """
    def __init__(self, n_joints: int, hidden_dim: int = 256):
        super().__init__()
        self.n_joints = n_joints
        # Input: joint angles (n_joints) + desired dx,dy (2)
        self.net = nn.Sequential(
            nn.Linear(n_joints + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_joints),
            nn.Tanh()           # output in [-1,1] matching action space
        )

    def forward(self, joint_angles: torch.Tensor, desired_disp: torch.Tensor) -> torch.Tensor:
        """
        joint_angles: (..., n_joints) normalized in [-1,1] (actual θ/π)
        desired_disp: (..., 2) desired end‑effector displacement (dx, dy) normalized in [-1,1]
        Returns: joint action (Δθ) normalized in [-1,1]
        """
        x = torch.cat([joint_angles, desired_disp], dim=-1)
        return self.net(x)


def train_action_mapper(mapper: ActionMapper,
                        joint_angles: np.ndarray,      # (N, n_joints) normalized
                        desired_disp: np.ndarray,      # (N, 2) normalized
                        target_actions: np.ndarray,    # (N, n_joints) normalized joint actions
                        epochs: int = 200,
                        batch_size: int = 256,
                        lr: float = 1e-3,
                        device: str = "cpu") -> ActionMapper:
    """Train a single action mapper using MSE loss."""
    mapper.to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(joint_angles, dtype=torch.float32),
        torch.tensor(desired_disp, dtype=torch.float32),
        torch.tensor(target_actions, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for ja, dd, ta in loader:
            ja, dd, ta = ja.to(device), dd.to(device), ta.to(device)
            pred = mapper(ja, dd)
            loss = criterion(pred, ta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} – Loss: {total_loss/len(loader):.6f}")
    return mapper
