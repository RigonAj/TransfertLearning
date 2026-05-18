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
# State Mapper (3 → 2)  ← CHANGEMENT PRINCIPAL
# ============================================================================
class StateMapperMLP(nn.Module):
    def __init__(self, input_dim: int = ARM_OBS_3DOF, output_dim: int = ARM_OBS_2DOF,
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
# Action Mapper (3-DoF state + 2-DoF action → 3-DoF action)
# ============================================================================
class ActionMapperMLP(nn.Module):
    def __init__(self, state_dim: int = ARM_OBS_3DOF, action_2dof_dim: int = 2,
                 output_dim: int = 3, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_2dof_dim, hidden),
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

    def forward(self, state_3dof: torch.Tensor, action_2dof: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state_3dof, action_2dof], dim=-1))


# ============================================================================
# Helper EE (inchangé)
# ============================================================================
def ee_from_state(state: np.ndarray, dof: int) -> np.ndarray:
    if dof == 2:
        theta1 = state[0]
        theta2 = state[1]
        l1, l2 = 1.5, 1.5
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    else:  # 3 dof
        theta1 = state[0]
        theta2 = state[1]
        theta3 = state[2]
        l1, l2, l3 = 1.0, 1.0, 1.0
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)
    return np.array([x, y])


# ============================================================================
# State Mapper Trainer (3 → 2)
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
        print("\nTraining State Mapper (3-DoF obs → 2-DoF obs)")
        print(f"  Input : {ARM_OBS_3DOF}D   Output : {ARM_OBS_2DOF}D")

        s3 = torch.tensor(trajectories['states_3dof'], dtype=torch.float32).to(self.device)
        s2 = torch.tensor(trajectories['states_2dof'], dtype=torch.float32).to(self.device)

        min_len = min(len(s3), len(s2))
        s3, s2 = s3[:min_len], s2[:min_len]

        # Split train/val
        n = len(s3)
        idx = torch.randperm(n)
        cut = int(n * 0.9)
        tr_idx, vl_idx = idx[:cut], idx[cut:]

        train_loader = DataLoader(TensorDataset(s3[tr_idx], s2[tr_idx]),
                                  batch_size=batch_size, shuffle=True, pin_memory=(self.device == "cuda"))
        val_loader = DataLoader(TensorDataset(s3[vl_idx], s2[vl_idx]),
                                batch_size=batch_size*2, shuffle=False, pin_memory=(self.device == "cuda"))

        # ... (le reste du training est identique à ta version 3to2, je le garde compact)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                         patience=15, min_lr=1e-6)
        best_val = float('inf')
        best_state = None
        wait = 0

        pbar = tqdm(range(epochs), desc="State Mapper (3→2)")
        for epoch in pbar:
            # training loop (identique à ton code existant)
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

            # validation
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
        print(f"  State Mapper (3→2) done - best val loss: {best_val:.6f}")


# ============================================================================
# Action Mapper Trainer (3obs + 2act → 3act)
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
        print("\nTraining Action Mapper (3-DoF state + 2-DoF action → 3-DoF action)")

        s3 = torch.tensor(trajectories['states_3dof'], dtype=torch.float32).to(self.device)
        a2 = torch.tensor(trajectories['actions_2dof'], dtype=torch.float32).to(self.device)
        a3 = torch.tensor(trajectories['actions_3dof'], dtype=torch.float32).to(self.device)

        min_len = min(len(s3), len(a2), len(a3))
        s3 = s3[:min_len]
        a2 = a2[:min_len]
        a3 = a3[:min_len]

        n = len(s3)
        idx = torch.randperm(n)
        cut = int(n * 0.9)
        tr_idx, vl_idx = idx[:cut], idx[cut:]

        train_loader = DataLoader(TensorDataset(s3[tr_idx], a2[tr_idx], a3[tr_idx]),
                                  batch_size=batch_size, shuffle=True, pin_memory=(self.device == "cuda"))
        val_loader = DataLoader(TensorDataset(s3[vl_idx], a2[vl_idx], a3[vl_idx]),
                                batch_size=batch_size*2, shuffle=False, pin_memory=(self.device == "cuda"))

        # ... training loop identique (je te laisse le tien si tu veux le copier-coller)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                         patience=15, min_lr=1e-6)
        best_val = float('inf')
        best_state = None
        wait = 0

        pbar = tqdm(range(epochs), desc="Action Mapper")
        for epoch in pbar:
            self.model.train()
            train_loss = 0.0
            for sb, a2b, a3b in train_loader:
                sb = sb.to(self.device, non_blocking=True)
                a2b = a2b.to(self.device, non_blocking=True)
                a3b = a3b.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                pred = self.model(sb, a2b)
                loss = self.criterion(pred, a3b)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for sb, a2b, a3b in val_loader:
                    sb = sb.to(self.device, non_blocking=True)
                    a2b = a2b.to(self.device, non_blocking=True)
                    a3b = a3b.to(self.device, non_blocking=True)
                    pred = self.model(sb, a2b)
                    loss = self.criterion(pred, a3b)
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
        print(f"  Action Mapper done - best val loss: {best_val:.6f}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path("./data/DIRECT")
    traj_path = data_dir / "trajectories_aligned.pkl"
    state_mapper_path = data_dir / "state_mapper_3to2_for_2to3.pt"      # nom clair
    action_mapper_path = data_dir / "action_mapper_3obs_2act_to_3act.pt"

    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)

    print(f"Loaded {trajectories['metadata']['n_samples']} samples")

    # State Mapper 3→2
    st = StateMapperTrainer(device=device, log_dir="./data/DIRECT/mappers_logs/state_3to2")
    st.train(trajectories)
    st.save(str(state_mapper_path))

    # Action Mapper 3+2→3
    at = ActionMapperTrainer(device=device, log_dir="./data/DIRECT/mappers_logs/action_3obs2act")
    at.train(trajectories)
    at.save(str(action_mapper_path))

    print("\n=== MAPPERS 2→3 TRANSFER READY ===")
    print(f"State  : {state_mapper_path}")
    print(f"Action : {action_mapper_path}")

if __name__ == "__main__":
    main()
