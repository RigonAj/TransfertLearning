import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


ARM_OBS_2DOF = 6
ARM_OBS_3DOF = 8
ACTION_DIM_2DOF = 2
ACTION_DIM_3DOF = 3
MAX_REACH = 3.0
LATENT_PUSHBALL_DIM = 18


class StateMapperMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionMapperMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_in_dim: int,
        act_out_dim: int,
        hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_in_dim = act_in_dim
        self.act_out_dim = act_out_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, act_out_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, act_in: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if act_in.dim() == 1:
            act_in = act_in.unsqueeze(0)
        return self.net(torch.cat([state, act_in], dim=-1))


def angle_slice_for_arm_dim(arm_dim: int) -> slice:
    return slice(0, 2) if arm_dim == ARM_OBS_2DOF else slice(0, 3)


def velocity_slice_for_arm_dim(arm_dim: int) -> slice:
    return slice(2, 4) if arm_dim == ARM_OBS_2DOF else slice(3, 6)


def eff_slice_for_arm_dim(arm_dim: int) -> slice:
    return slice(4, 6) if arm_dim == ARM_OBS_2DOF else slice(6, 8)


def dof_from_arm_dim(arm_dim: int) -> int:
    if arm_dim == ARM_OBS_2DOF:
        return 2
    if arm_dim == ARM_OBS_3DOF:
        return 3
    raise ValueError(f"Unsupported arm observation dim: {arm_dim}")


def _wrap_pi_np(angles: np.ndarray) -> np.ndarray:
    return ((angles + np.pi) % (2.0 * np.pi)) - np.pi


def _fk_np(angles: np.ndarray) -> np.ndarray:
    dof = angles.shape[-1]
    if dof == 2:
        t1 = angles[..., 0]
        t2 = angles[..., 1]
        x = 1.5 * np.cos(t1) + 1.5 * np.cos(t1 + t2)
        y = 1.5 * np.sin(t1) + 1.5 * np.sin(t1 + t2)
    elif dof == 3:
        t1 = angles[..., 0]
        t2 = angles[..., 1]
        t3 = angles[..., 2]
        x = np.cos(t1) + np.cos(t1 + t2) + np.cos(t1 + t2 + t3)
        y = np.sin(t1) + np.sin(t1 + t2) + np.sin(t1 + t2 + t3)
    else:
        raise ValueError(f"Unsupported dof: {dof}")
    return np.stack([x, y], axis=-1).astype(np.float32)


def fk_from_arm_state_torch(arm_state: torch.Tensor) -> torch.Tensor:
    arm_dim = arm_state.shape[-1]
    if arm_dim == ARM_OBS_2DOF:
        t1 = arm_state[..., 0] * torch.pi
        t2 = arm_state[..., 1] * torch.pi
        x = 1.5 * torch.cos(t1) + 1.5 * torch.cos(t1 + t2)
        y = 1.5 * torch.sin(t1) + 1.5 * torch.sin(t1 + t2)
    elif arm_dim == ARM_OBS_3DOF:
        t1 = arm_state[..., 0] * torch.pi
        t2 = arm_state[..., 1] * torch.pi
        t3 = arm_state[..., 2] * torch.pi
        x = torch.cos(t1) + torch.cos(t1 + t2) + torch.cos(t1 + t2 + t3)
        y = torch.sin(t1) + torch.sin(t1 + t2) + torch.sin(t1 + t2 + t3)
    else:
        raise ValueError(f"Unsupported arm observation dim: {arm_dim}")
    return torch.stack([x, y], dim=-1) / MAX_REACH


def arm_end_effector_np(arm_state: np.ndarray) -> np.ndarray:
    arm_state = np.asarray(arm_state, dtype=np.float32)
    return arm_state[..., eff_slice_for_arm_dim(arm_state.shape[-1])] * MAX_REACH


def angular_mse(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    pred = pred_norm * torch.pi
    target = target_norm * torch.pi
    diff = torch.atan2(torch.sin(pred - target), torch.cos(pred - target))
    return torch.mean(diff * diff)


def _project_angles_to_ee(
    angles: np.ndarray,
    desired_ee: np.ndarray,
    damping: float = 0.08,
    max_iters: int = 12,
    max_delta: float = 0.35,
) -> np.ndarray:
    q = _wrap_pi_np(angles.astype(np.float32))
    target = desired_ee.astype(np.float32).copy()
    norm = float(np.linalg.norm(target))
    if norm > MAX_REACH - 1e-4:
        target *= (MAX_REACH - 1e-4) / norm

    if q.shape[-1] == 2:
        x, y = float(target[0]), float(target[1])
        l1 = 1.5
        l2 = 1.5
        r = float(np.hypot(x, y))
        reach = l1 + l2 - 1e-6
        if r > reach:
            x *= reach / r
            y *= reach / r
            r = reach
        cos_t2 = (r * r - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_t2 = float(np.clip(cos_t2, -1.0, 1.0))
        t2 = float(np.arccos(cos_t2))
        k1 = l1 + l2 * cos_t2
        k2 = l2 * np.sin(t2)
        t1 = float(np.arctan2(y, x) - np.arctan2(k2, k1))
        return _wrap_pi_np(np.array([t1, t2], dtype=np.float32)).astype(np.float32)

    for _ in range(max_iters):
        err = target - _fk_np(q)
        if float(np.linalg.norm(err)) < 1e-4:
            break
        j = np.empty((2, 3), dtype=np.float32)
        a = q[0]
        b = q[0] + q[1]
        c = q[0] + q[1] + q[2]
        j[0] = [-np.sin(a) - np.sin(b) - np.sin(c), -np.sin(b) - np.sin(c), -np.sin(c)]
        j[1] = [np.cos(a) + np.cos(b) + np.cos(c), np.cos(b) + np.cos(c), np.cos(c)]
        dq = j.T @ np.linalg.solve(j @ j.T + (damping ** 2) * np.eye(2, dtype=np.float32), err)
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_delta:
            dq *= max_delta / dq_norm
        q = _wrap_pi_np(q + dq)
    return q.astype(np.float32)


def project_mapped_arm_state_to_reference(
    mapped_arm_state: np.ndarray,
    reference_arm_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    mapped = np.asarray(mapped_arm_state, dtype=np.float32)
    single = mapped.ndim == 1
    if single:
        mapped = mapped.reshape(1, -1)

    if reference_arm_state is None:
        desired_ee = arm_end_effector_np(mapped)
    else:
        ref = np.asarray(reference_arm_state, dtype=np.float32)
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        desired_ee = arm_end_effector_np(ref)

    out = np.clip(mapped.copy(), -1.0, 1.0)
    arm_dim = out.shape[-1]
    angle_sl = angle_slice_for_arm_dim(arm_dim)
    velocity_sl = velocity_slice_for_arm_dim(arm_dim)
    eff_sl = eff_slice_for_arm_dim(arm_dim)

    for i in range(out.shape[0]):
        q0 = out[i, angle_sl] * np.pi
        q = _project_angles_to_ee(q0, desired_ee[i])
        out[i, angle_sl] = q / np.pi
        out[i, velocity_sl] = np.clip(out[i, velocity_sl], -1.0, 1.0)
        out[i, eff_sl] = np.clip(_fk_np(q) / MAX_REACH, -1.0, 1.0)

    return out[0] if single else out


def load_state_mapper(input_dim: int, output_dim: int, path, device: str = "cpu") -> Optional[StateMapperMLP]:
    if path is None or not os.path.exists(path):
        return None
    mapper = StateMapperMLP(input_dim, output_dim).to(device)
    mapper.load_state_dict(torch.load(path, map_location=device))
    mapper.eval()
    return mapper


def load_action_mapper(state_dim: int, act_in_dim: int, act_out_dim: int, path, device: str = "cpu") -> Optional[ActionMapperMLP]:
    if path is None or not os.path.exists(path):
        return None
    mapper = ActionMapperMLP(state_dim, act_in_dim, act_out_dim).to(device)
    mapper.load_state_dict(torch.load(path, map_location=device))
    mapper.eval()
    return mapper


def save_latent_mappers(
    save_dir: str,
    state_2to3: StateMapperMLP,
    state_3to2: StateMapperMLP,
    action_2to3: ActionMapperMLP,
    action_3to2: ActionMapperMLP,
    run_id: str = "latent_reaching",
) -> Path:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_2to3": state_2to3.state_dict(),
        "state_3to2": state_3to2.state_dict(),
        "action_2to3": action_2to3.state_dict(),
        "action_3to2": action_3to2.state_dict(),
        "config": {
            "ARM_OBS_2DOF": ARM_OBS_2DOF,
            "ARM_OBS_3DOF": ARM_OBS_3DOF,
            "ACTION_DIM_2DOF": ACTION_DIM_2DOF,
            "ACTION_DIM_3DOF": ACTION_DIM_3DOF,
            "LATENT_PUSHBALL_DIM": LATENT_PUSHBALL_DIM,
        },
    }
    torch.save(checkpoint, save_path / f"{run_id}_mappers.pt")

    torch.save(state_2to3.state_dict(), save_path / "state_mapper_2to3.pt")
    torch.save(state_3to2.state_dict(), save_path / "state_mapper_3to2.pt")
    torch.save(action_2to3.state_dict(), save_path / "action_mapper_2to3.pt")
    torch.save(action_3to2.state_dict(), save_path / "action_mapper_3to2.pt")
    return save_path


def load_latent_mappers(
    save_dir: str,
    device: str = "cpu",
    hidden: int = 512,
) -> Tuple[StateMapperMLP, StateMapperMLP, ActionMapperMLP, ActionMapperMLP]:
    save_path = Path(save_dir)
    checkpoint_path = save_path / "latent_reaching_mappers.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_2to3 = StateMapperMLP(ARM_OBS_2DOF, ARM_OBS_3DOF, hidden=hidden).to(device)
        state_3to2 = StateMapperMLP(ARM_OBS_3DOF, ARM_OBS_2DOF, hidden=hidden).to(device)
        action_2to3 = ActionMapperMLP(ARM_OBS_3DOF, ACTION_DIM_2DOF, ACTION_DIM_3DOF, hidden=hidden).to(device)
        action_3to2 = ActionMapperMLP(ARM_OBS_2DOF, ACTION_DIM_3DOF, ACTION_DIM_2DOF, hidden=hidden).to(device)
        state_2to3.load_state_dict(checkpoint["state_2to3"])
        state_3to2.load_state_dict(checkpoint["state_3to2"])
        action_2to3.load_state_dict(checkpoint["action_2to3"])
        action_3to2.load_state_dict(checkpoint["action_3to2"])
    else:
        state_2to3 = load_state_mapper(ARM_OBS_2DOF, ARM_OBS_3DOF, save_path / "state_mapper_2to3.pt", device)
        state_3to2 = load_state_mapper(ARM_OBS_3DOF, ARM_OBS_2DOF, save_path / "state_mapper_3to2.pt", device)
        action_2to3 = load_action_mapper(ARM_OBS_3DOF, ACTION_DIM_2DOF, ACTION_DIM_3DOF, save_path / "action_mapper_2to3.pt", device)
        action_3to2 = load_action_mapper(ARM_OBS_2DOF, ACTION_DIM_3DOF, ACTION_DIM_2DOF, save_path / "action_mapper_3to2.pt", device)
        if any(m is None for m in (state_2to3, state_3to2, action_2to3, action_3to2)):
            raise FileNotFoundError(f"Missing mapper checkpoint in {save_dir}")

    for mapper in (state_2to3, state_3to2, action_2to3, action_3to2):
        mapper.eval()
    return state_2to3, state_3to2, action_2to3, action_3to2
