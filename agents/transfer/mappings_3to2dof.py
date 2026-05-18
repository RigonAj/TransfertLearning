import os
import gc
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(4)

ARM_OBS_2DOF = 6
ARM_OBS_3DOF = 8

# ============================================================================
# State Mapper (2 → 3)
# ============================================================================
class StateMapperMLP(nn.Module):
    def __init__(self, input_dim: int = ARM_OBS_2DOF, output_dim: int = ARM_OBS_3DOF,
                 hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# Action Mapper (2-DoF state + 3-DoF action → 2-DoF action)
# ============================================================================
class ActionMapperMLP(nn.Module):
    def __init__(self, state_dim: int = ARM_OBS_2DOF, action_3dof_dim: int = 3,
                 output_dim: int = 2, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_3dof_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
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

    def forward(self, state_2dof: torch.Tensor, action_3dof: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state_2dof, action_3dof], dim=-1))


# ============================================================================
# Helper: calcul de l'effecteur
# ============================================================================
def ee_from_state(state: np.ndarray, dof: int) -> np.ndarray:
    if dof == 2:
        theta1 = state[0]
        theta2 = state[1]
        l1, l2 = 1.5, 1.5
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    else:  # dof == 3
        theta1 = state[0]
        theta2 = state[1]
        theta3 = state[2]
        l1, l2, l3 = 1.0, 1.0, 1.0
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)
    return np.array([x, y])


# ============================================================================
# State Mapper Trainer (2 → 3)
# ============================================================================
class StateMapperTrainer:
    def __init__(self, device: str = "cpu", lr: float = 3e-4, log_dir: str = None):
        self.device = device
        self.model = StateMapperMLP().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter(log_dir) if log_dir else None

    def train(self, trajectories: Dict, epochs: int = 500, batch_size: int = 512,
              patience: int = 35, min_delta: float = 1e-5):
        print("\nTraining State Mapper (2-DoF → 3-DoF) for 3→2 transfer")
        print(f"  Input : {ARM_OBS_2DOF}D   Output : {ARM_OBS_3DOF}D")

        s2 = torch.tensor(trajectories['states_2dof'], dtype=torch.float32).to(self.device)
        s3 = torch.tensor(trajectories['states_3dof'], dtype=torch.float32).to(self.device)

        min_len = min(len(s2), len(s3))
        s2, s3 = s2[:min_len], s3[:min_len]

        n = len(s2)
        idx = torch.randperm(n)
        cut = int(n * 0.9)
        tr_idx, vl_idx = idx[:cut], idx[cut:]

        train_loader = DataLoader(TensorDataset(s2[tr_idx], s3[tr_idx]),
                                  batch_size=batch_size, shuffle=True,
                                  pin_memory=(self.device == "cuda"))
        val_loader = DataLoader(TensorDataset(s2[vl_idx], s3[vl_idx]),
                                batch_size=batch_size*2, shuffle=False,
                                pin_memory=(self.device == "cuda"))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                         factor=0.5, patience=15, min_lr=1e-6)
        best_val = float('inf')
        best_state = None
        wait = 0

        pbar = tqdm(range(epochs), desc="State Mapper (2→3)")
        for epoch in pbar:
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    loss = self.criterion(self.model(xb), yb)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

            pbar.set_postfix({"train": f"{train_loss:.5f}", "val": f"{val_loss:.5f}"})

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"  State Mapper (2→3) complete - best val loss: {best_val:.6f}")


# ============================================================================
# Action Mapper Trainer (2-DoF state + 3-DoF action → 2-DoF action)
# ============================================================================
class ActionMapperTrainer:
    def __init__(self, device: str = "cpu", lr: float = 3e-4, log_dir: str = None):
        self.device = device
        self.model = ActionMapperMLP().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter(log_dir) if log_dir else None

    def train(self, trajectories: Dict, epochs: int = 500, batch_size: int = 512,
              patience: int = 35, min_delta: float = 1e-5):
        print("\nTraining Action Mapper (2-DoF state + 3-DoF action → 2-DoF action)")

        s2 = torch.tensor(trajectories['states_2dof'], dtype=torch.float32).to(self.device)
        a3 = torch.tensor(trajectories['actions_3dof'], dtype=torch.float32).to(self.device)
        a2 = torch.tensor(trajectories['actions_2dof'], dtype=torch.float32).to(self.device)

        min_len = min(len(s2), len(a3), len(a2))
        s2 = s2[:min_len]
        a3 = a3[:min_len]
        a2 = a2[:min_len]

        n = len(s2)
        idx = torch.randperm(n)
        cut = int(n * 0.9)
        tr_idx, vl_idx = idx[:cut], idx[cut:]

        train_loader = DataLoader(TensorDataset(s2[tr_idx], a3[tr_idx], a2[tr_idx]),
                                  batch_size=batch_size, shuffle=True,
                                  pin_memory=(self.device == "cuda"))
        val_loader = DataLoader(TensorDataset(s2[vl_idx], a3[vl_idx], a2[vl_idx]),
                                batch_size=batch_size*2, shuffle=False,
                                pin_memory=(self.device == "cuda"))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                         factor=0.5, patience=15, min_lr=1e-6)
        best_val = float('inf')
        best_state = None
        wait = 0

        pbar = tqdm(range(epochs), desc="Action Mapper")
        for epoch in pbar:
            self.model.train()
            train_loss = 0.0
            for sb, a3b, a2b in train_loader:
                sb = sb.to(self.device, non_blocking=True)
                a3b = a3b.to(self.device, non_blocking=True)
                a2b = a2b.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                pred = self.model(sb, a3b)
                loss = self.criterion(pred, a2b)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for sb, a3b, a2b in val_loader:
                    sb = sb.to(self.device, non_blocking=True)
                    a3b = a3b.to(self.device, non_blocking=True)
                    a2b = a2b.to(self.device, non_blocking=True)
                    pred = self.model(sb, a3b)
                    loss = self.criterion(pred, a2b)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

            pbar.set_postfix({"train": f"{train_loss:.5f}", "val": f"{val_loss:.5f}"})

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"  Action Mapper complete - best val loss: {best_val:.6f}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path("./data/DIRECT")
    traj_path = data_dir / "trajectories_aligned.pkl"

    state_mapper_path = data_dir / "state_mapper_2to3_for_3to2.pt"
    action_mapper_path = data_dir / "action_mapper_2state_3act_to_2act.pt"

    print("\n" + "="*70)
    print("LOADING TRAJECTORIES")
    print("="*70)
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"  {trajectories['metadata']['n_samples']} samples from {trajectories['metadata']['n_segments']} segments")

    # State Mapper
    print("\n" + "="*70)
    print("STATE MAPPER (2 → 3)")
    print("="*70)
    st = StateMapperTrainer(device=device, log_dir="./data/DIRECT/mappers_logs/state_2to3_3to2")
    st.train(trajectories)
    st.save(str(state_mapper_path))

    # Action Mapper
    print("\n" + "="*70)
    print("ACTION MAPPER (2-DoF state + 3-DoF action → 2-DoF action)")
    print("="*70)
    at = ActionMapperTrainer(device=device, log_dir="./data/DIRECT/mappers_logs/action_2state3act_to_2act")
    at.train(trajectories)
    at.save(str(action_mapper_path))

    print("\n" + "="*70)
    print("MAPPER TRAINING 3→2 COMPLETE")
    print("="*70)
    print(f"State Mapper  → {state_mapper_path}")
    print(f"Action Mapper → {action_mapper_path}")


if __name__ == "__main__":
    main()
