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
# Action Mapper (state_2dof + action_2dof → action_3dof)
# ============================================================================
class ActionMapperMLP(nn.Module):
    def __init__(self, state_dim: int = ARM_OBS_2DOF, action_2dof_dim: int = 2,
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

    def forward(self, state_2dof: torch.Tensor, action_2dof: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state_2dof, action_2dof], dim=-1))

# ============================================================================
# Helper: calcul de l'effecteur (version corrigée avec les vraies longueurs)
# ============================================================================
def ee_from_state(state: np.ndarray, dof: int) -> np.ndarray:
    if dof == 2:
        theta1 = state[0]
        theta2 = state[1]
        l1, l2 = 1.5, 1.5   # longueurs réelles du bras 2-DoF
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    else:  # dof == 3
        theta1 = state[0]
        theta2 = state[1]
        theta3 = state[2]
        l1, l2, l3 = 1.0, 1.0, 1.0   # longueurs réelles du bras 3-DoF
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)
    return np.array([x, y])

# ============================================================================
# State Mapper Trainer
# ============================================================================
class StateMapperTrainer:
    def __init__(self, device: str = "cpu", lr: float = 3e-4, log_dir: str = None):
        self.device = device
        self.model = StateMapperMLP().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter(log_dir) if log_dir else None

    def train(self, trajectories: Dict, epochs: int = 500, batch_size: int = 512,
              patience: int = 30, min_delta: float = 1e-5):
        print("\nTraining State Mapper (2-DoF → 3-DoF)")
        print(f"  Input : {ARM_OBS_2DOF}D   Output : {ARM_OBS_3DOF}D")
        if self.writer:
            print(f"  TensorBoard logs -> {self.writer.log_dir}")

        s2 = torch.tensor(trajectories['states_2dof'], dtype=torch.float32).to(self.device)
        s3 = torch.tensor(trajectories['states_3dof'], dtype=torch.float32).to(self.device)

        # Optionnel : si les tailles sont inégales (à cause d'un décalage), on aligne
        min_len = min(len(s2), len(s3))
        s2, s3 = s2[:min_len], s3[:min_len]

        n = len(s2)
        idx = torch.randperm(n)
        cut = int(n * 0.9)
        tr_idx, vl_idx = idx[:cut], idx[cut:]

        s2_tr, s3_tr = s2[tr_idx], s3[tr_idx]
        s2_vl, s3_vl = s2[vl_idx], s3[vl_idx]

        train_loader = DataLoader(TensorDataset(s2_tr, s3_tr), batch_size=batch_size, shuffle=True,
                                  pin_memory=(self.device == "cuda"), drop_last=False)
        val_loader = DataLoader(TensorDataset(s2_vl, s3_vl), batch_size=batch_size*2, shuffle=False,
                                pin_memory=(self.device == "cuda"))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                         patience=15, min_lr=1e-6)
        best_val = float('inf')
        best_state = None
        wait = 0

        # Barre de progression sur les epochs
        pbar_epoch = tqdm(range(epochs), desc="Epochs")
        for epoch in pbar_epoch:
            epoch_start = time.time()
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
            current_lr = self.optimizer.param_groups[0]['lr']

            # Early stopping
            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    pbar_epoch.set_postfix({"early_stop": epoch+1})
                    break

            # Logs TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Learning_rate", current_lr, epoch)
                self.writer.add_scalar("Time/epoch_seconds", time.time() - epoch_start, epoch)

                if (epoch+1) % 10 == 0:
                    x_sample, y_true = next(iter(val_loader))
                    x_sample = x_sample[:32].to(self.device)
                    y_pred = self.model(x_sample).detach().cpu().numpy()
                    y_true_np = y_true[:32].cpu().numpy()
                    ee_err = 0.0
                    for i in range(len(x_sample)):
                        ee_true = ee_from_state(y_true_np[i], 3)
                        ee_pred = ee_from_state(y_pred[i], 3)
                        ee_err += np.linalg.norm(ee_true - ee_pred)
                    ee_err /= len(x_sample)
                    self.writer.add_scalar("Metrics/ee_error_val", ee_err, epoch)

            # Mise à jour de la barre
            pbar_epoch.set_postfix({
                "train": f"{train_loss:.5f}",
                "val": f"{val_loss:.5f}"
            })

            # Nettoyage mémoire
            if (epoch+1) % 20 == 0:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"  State Mapper training complete (best val loss: {best_val:.6f})")
        if self.writer:
            self.writer.close()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"State Mapper saved → {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"State Mapper loaded ← {path}")

# ============================================================================
# Action Mapper Trainer
# ============================================================================
class ActionMapperTrainer:
    def __init__(self, device: str = "cpu", lr: float = 3e-4, log_dir: str = None):
        self.device = device
        self.model = ActionMapperMLP().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter(log_dir) if log_dir else None

    def train(self, trajectories: Dict, epochs: int = 500, batch_size: int = 512,
              patience: int = 30, min_delta: float = 1e-5):
        print("\nTraining Action Mapper (state_2dof + action_2dof → action_3dof)")
        print(f"  Input : {ARM_OBS_2DOF + 2}D   Output : 3D")
        if self.writer:
            print(f"  TensorBoard logs -> {self.writer.log_dir}")

        s2 = torch.tensor(trajectories['states_2dof'], dtype=torch.float32).to(self.device)
        a2 = torch.tensor(trajectories['actions_2dof'], dtype=torch.float32).to(self.device)
        a3 = torch.tensor(trajectories['actions_3dof'], dtype=torch.float32).to(self.device)

        # Alignement des longueurs (les actions ont un élément de moins que les états)
        min_len = min(len(s2), len(a2), len(a3))
        s2 = s2[:min_len]
        a2 = a2[:min_len]
        a3 = a3[:min_len]

        n = len(s2)
        idx = torch.randperm(n)
        cut = int(n * 0.9)
        tr_idx, vl_idx = idx[:cut], idx[cut:]

        s2_tr, a2_tr, a3_tr = s2[tr_idx], a2[tr_idx], a3[tr_idx]
        s2_vl, a2_vl, a3_vl = s2[vl_idx], a2[vl_idx], a3[vl_idx]

        train_loader = DataLoader(TensorDataset(s2_tr, a2_tr, a3_tr), batch_size=batch_size, shuffle=True,
                                  pin_memory=(self.device == "cuda"), drop_last=False)
        val_loader = DataLoader(TensorDataset(s2_vl, a2_vl, a3_vl), batch_size=batch_size*2, shuffle=False,
                                pin_memory=(self.device == "cuda"))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                         patience=15, min_lr=1e-6)
        best_val = float('inf')
        best_state = None
        wait = 0

        pbar_epoch = tqdm(range(epochs), desc="Epochs")
        for epoch in pbar_epoch:
            epoch_start = time.time()
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
            current_lr = self.optimizer.param_groups[0]['lr']

            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    pbar_epoch.set_postfix({"early_stop": epoch+1})
                    break

            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Learning_rate", current_lr, epoch)
                self.writer.add_scalar("Time/epoch_seconds", time.time() - epoch_start, epoch)

            pbar_epoch.set_postfix({
                "train": f"{train_loss:.5f}",
                "val": f"{val_loss:.5f}"
            })

            if (epoch+1) % 20 == 0:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"  Action Mapper training complete (best val loss: {best_val:.6f})")
        if self.writer:
            self.writer.close()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Action Mapper saved → {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Action Mapper loaded ← {path}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 500
    patience = 35

    data_dir = Path("./data/DIRECT")
    traj_path = data_dir / "trajectories_aligned.pkl"
    state_mapper_path = data_dir / "state_mapper_2to3dof.pt"
    action_mapper_path = data_dir / "action_mapper_2to3dof.pt"

    print("\n" + "="*60)
    print("LOADING TRAJECTORIES")
    print("="*60)
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"  {trajectories['metadata']['n_samples']} samples from {trajectories['metadata']['n_segments']} segments")

    print("\n" + "="*60)
    print("STATE MAPPER (2 → 3)")
    print("="*60)
    st = StateMapperTrainer(device=device, log_dir="./data/DIRECT/mappers_logs/state_mapper_2to3dof")
    st.train(trajectories, epochs=epochs, batch_size=512, patience=patience)
    st.save(str(state_mapper_path))

    print("\n" + "="*60)
    print("ACTION MAPPER (2-DoF state + 2-DoF action → 3-DoF action)")
    print("="*60)
    at = ActionMapperTrainer(device=device, log_dir="./data/DIRECT/mappers_logs/action_mapper_2to3dof")
    at.train(trajectories, epochs=epochs, batch_size=512, patience=patience)
    at.save(str(action_mapper_path))

    print("\n" + "="*60)
    print("MAPPER TRAINING COMPLETE")
    print("="*60)
    print(f"  State Mapper  → {state_mapper_path}")
    print(f"  Action Mapper → {action_mapper_path}")
    print("\nTo view TensorBoard, run: tensorboard --logdir=data.DIRECT.mappers_logs")

if __name__ == "__main__":
    main()
