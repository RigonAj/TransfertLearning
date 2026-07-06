"""
UNN Bases — Fixed Cartesian latent space and analytic IK mapping.

This module provides:
- `CartesianStateEncoder`: converts raw observations (arm + task) into a
    fixed Cartesian representation (effector pos + environment) normalized
    by the robot `max_reach`.
- Analytic kinematics helpers (`planar_jacobian`, `ik_damped_least_squares`,
    `ik_velocity`) used to convert desired end-effector velocities into joint
    velocities (rad/s).

Pipeline (velocity-based):
    PPO predicts (vx, vy) normalized in [-1,1]
    → v_ee = latent_action * v_max_ee          (m/s)
    → omega_joints = ik_velocity(q, v_ee, L)   (rad/s)  via J^+
    → joint_action = omega_joints / omega_max  (normalized, sent to env)
    → env applies: new_theta = theta + joint_action * omega_max * dt
"""

import torch
import numpy as np
from typing import Tuple
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------

class CartesianStateEncoder:
    """
    Encode raw observation (arm + task) into a fixed Cartesian latent state.
    For PushBall: [eff_x, eff_y, ball_x, ball_y, tgt_x, tgt_y] normalized by max_reach.

    Note: max_reach is used ONLY for observation normalization.
    Action scaling uses v_max_ee (stored in UNNPolicy / UNNLatentEnv).
    """
    def __init__(self, max_reach: float = 3.0):
        self.max_reach = max_reach

    def encode(self, obs: np.ndarray, arm_obs_size: int) -> np.ndarray:
        env_part, eff_pos = split_env_and_effector(obs, arm_obs_size, self.max_reach)
        latent = np.concatenate([eff_pos, env_part])  # [eff_x, eff_y, ball_x, ball_y, tgt_x, tgt_y]
        return np.clip(latent, -1.0, 1.0)


def split_env_and_effector(obs: np.ndarray, arm_obs_size: int, max_reach: float = 3.0):
    """
    Split a raw observation into environment part and effector position.

    Returns two arrays normalized by max_reach:
      - env_part: 4D [ball_x, ball_y, tgt_x, tgt_y] / max_reach
      - eff_pos:  2D [eff_x, eff_y] / max_reach
    """
    obs = np.asarray(obs, dtype=np.float32)
    if arm_obs_size == 6:
        eff_x = obs[4]
        eff_y = obs[5]
    else:
        eff_x = obs[6]
        eff_y = obs[7]

    env_part = obs[arm_obs_size:arm_obs_size + 4]
    eff_pos = np.array([eff_x, eff_y], dtype=np.float32)
    return env_part.astype(np.float32), eff_pos.astype(np.float32)


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def planar_jacobian(joint_angles: np.ndarray, link_lengths: np.ndarray) -> np.ndarray:
    """
    Compute the 2 x n Jacobian for a planar serial manipulator.

    joint_angles: (n,) in radians
    link_lengths: (n,)
    Returns: J (2, n)
    """
    joint_angles = np.asarray(joint_angles, dtype=np.float32)
    link_lengths = np.asarray(link_lengths, dtype=np.float32)
    n = joint_angles.shape[0]
    phi = np.cumsum(joint_angles)
    J = np.zeros((2, n), dtype=np.float32)
    for k in range(n):
        s = 0.0
        c = 0.0
        for i in range(k, n):
            s -= float(link_lengths[i]) * float(np.sin(phi[i]))
            c += float(link_lengths[i]) * float(np.cos(phi[i]))
        J[0, k] = s
        J[1, k] = c
    return J


def ik_damped_least_squares(joint_angles: np.ndarray,
                            delta_xy: np.ndarray,
                            link_lengths: np.ndarray,
                            damping: float = 1e-3) -> np.ndarray:
    """
    Compute joint increments (radians) for a small end-effector displacement
    using damped least squares (Levenberg-Marquardt style).

    joint_angles: (n,) radians
    delta_xy:     (2,) meters (dx, dy)
    link_lengths: (n,)
    damping:      float

    Returns: delta_theta (n,) radians
    """
    delta_xy = np.asarray(delta_xy, dtype=np.float32).reshape(2,)
    J    = planar_jacobian(joint_angles, link_lengths)
    JJt  = J @ J.T
    reg  = (damping ** 2) * np.eye(2, dtype=np.float32)
    inv  = np.linalg.inv(JJt + reg)
    dq   = J.T @ (inv @ delta_xy)
    return dq.astype(np.float32)


def ik_velocity(joint_angles: np.ndarray,
                v_ee: np.ndarray,
                link_lengths: np.ndarray,
                damping: float = 1e-3) -> np.ndarray:
    """
    Compute joint velocities (rad/s) from a desired end-effector velocity (m/s)
    using the damped pseudo-inverse Jacobian:

        omega_joints = J^T (J J^T + λ²I)^{-1} v_ee

    joint_angles: (n,) radians  — current configuration
    v_ee:         (2,) m/s      — desired [vx, vy]
    link_lengths: (n,)
    damping:      float

    Returns: omega_joints (n,) rad/s
    """
    v_ee = np.asarray(v_ee, dtype=np.float32).reshape(2,)
    J    = planar_jacobian(joint_angles, link_lengths)
    JJt  = J @ J.T
    reg  = (damping ** 2) * np.eye(2, dtype=np.float32)
    inv  = np.linalg.inv(JJt + reg)
    omega = J.T @ (inv @ v_ee)
    return omega.astype(np.float32)
