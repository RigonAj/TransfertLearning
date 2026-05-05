import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

torch.set_num_threads(8)

# ============================================================================
# Dimensions des observations bras (après refactoring envs) :
#   2-dof arm_obs : 6D  [θ1/π, θ2/π, dθ1/ω, dθ2/ω, eff_x/r, eff_y/r]
#   3-dof arm_obs : 8D  [θ1/π, θ2/π, θ3/π, dθ1/ω, dθ2/ω, dθ3/ω, eff_x/r, eff_y/r]
# ============================================================================

ARM_OBS_2dof = 6
ARM_OBS_3dof = 8


# ============================================================================
# PART 1: STATE MAPPER  2-dof arm_obs (6) → 3-dof arm_obs (8)
# ============================================================================

class StateMapperMLP(nn.Module):
    def __init__(self, input_dim: int = ARM_OBS_2dof, output_dim: int = ARM_OBS_3dof,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateMapperTrainer:
    def __init__(self, device: str = "cpu", lr: float = 3e-4):
        self.device = device
        self.model = StateMapperMLP().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def _split(self, x, y, val_frac=0.1):
        n = len(x)
        idx = torch.randperm(n)
        cut = int(n * (1 - val_frac))
        tr, vl = idx[:cut], idx[cut:]
        return x[tr], y[tr], x[vl], y[vl]

    def train(self, trajectories: Dict, epochs: int = 300, batch_size: int = 512):
        print("\nTraining State Mapper (2-dof arm_obs → 3-dof arm_obs)")
        print(f"  Input : {ARM_OBS_2dof}D   Output : {ARM_OBS_3dof}D")

        s2 = torch.tensor(
            np.concatenate(trajectories['states_2dof'], axis=0), dtype=torch.float32
        ).to(self.device)
        s3 = torch.tensor(
            np.concatenate(trajectories['states_3dof'], axis=0), dtype=torch.float32
        ).to(self.device)

        assert s2.shape[1] == ARM_OBS_2dof, \
            f"states_2dof : attendu {ARM_OBS_2dof}D, obtenu {s2.shape[1]}D"
        assert s3.shape[1] == ARM_OBS_3dof, \
            f"states_3dof : attendu {ARM_OBS_3dof}D, obtenu {s3.shape[1]}D"
        print(f"  Dataset : {len(s2):,} samples\n")

        x_tr, y_tr, x_vl, y_vl = self._split(s2, s3)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        best_val = float('inf')
        best_state = None

        pbar_epoch = tqdm(range(epochs), desc="Epochs")

        for epoch in pbar_epoch:
            self.model.train()
            perm = torch.randperm(len(x_tr))
            total_loss, n_b = 0.0, 0
            for i in range(0, len(x_tr), batch_size):
                idx = perm[i:i + batch_size]
                loss = self.criterion(self.model(x_tr[idx]), y_tr[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item(); n_b += 1
            scheduler.step()

            if (epoch + 1) % 25 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self.criterion(self.model(x_vl), y_vl).item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"  State Mapper training complete (best val loss: {best_val:.6f})")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"\nState Mapper saved → {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"\nState Mapper loaded ← {path}")


# ============================================================================
# PART 2: ACTION MAPPER  (2-dof arm_obs (6) + 3-dof action (3)) → 2-dof action (2)
# ============================================================================

class ActionMapperMLP(nn.Module):
    def __init__(self, state_dim: int = ARM_OBS_2dof, action_3dof_dim: int = 3,
                 output_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_3dof_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim),
            nn.Tanh(),   # actions bounded to [-1, 1]
        )

    def forward(self, state_2dof: torch.Tensor, action_3dof: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_2dof, action_3dof], dim=-1)
        return self.net(x)


class ActionMapperTrainer:
    def __init__(self, device: str = "cpu", lr: float = 3e-4):
        self.device = device
        self.model = ActionMapperMLP().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def _split(self, *tensors, val_frac=0.1):
        n = len(tensors[0])
        idx = torch.randperm(n)
        cut = int(n * (1 - val_frac))
        tr, vl = idx[:cut], idx[cut:]
        return tuple(t[tr] for t in tensors) + tuple(t[vl] for t in tensors)

    def train(self, trajectories: Dict, epochs: int = 300, batch_size: int = 512):
        print("\nTraining Action Mapper (arm_obs_2dof + action_3dof → action_2dof)")
        print(f"  Input : {ARM_OBS_2dof + 2}D   Output : 3D")

        s2 = torch.tensor(
            np.concatenate(trajectories['states_2dof'],  axis=0), dtype=torch.float32
        ).to(self.device)
        a3 = torch.tensor(
            np.concatenate(trajectories['actions_3dof'], axis=0), dtype=torch.float32
        ).to(self.device)
        a2 = torch.tensor(
            np.concatenate(trajectories['actions_2dof'], axis=0), dtype=torch.float32
        ).to(self.device)

        assert s2.shape[1] == ARM_OBS_2dof, \
            f"states_2dof : attendu {ARM_OBS_2dof}D, obtenu {s2.shape[1]}D"
        print(f"  Dataset : {len(s2):,} samples\n")

        splits = self._split(s2, a3, a2)
        s2_tr, a3_tr, a2_tr = splits[0], splits[1], splits[2]
        s2_vl, a3_vl, a2_vl = splits[3], splits[4], splits[5]

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        best_val = float('inf')
        best_state = None

        pbar_epoch = tqdm(range(epochs), desc="Epochs")

        for epoch in pbar_epoch:
            self.model.train()
            perm = torch.randperm(len(s2_tr))
            total_loss, n_b = 0.0, 0
            for i in range(0, len(s2_tr), batch_size):
                idx = perm[i:i + batch_size]
                pred = self.model(s2_tr[idx], a3_tr[idx])
                loss = self.criterion(pred, a2_tr[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item(); n_b += 1
            scheduler.step()

            if (epoch + 1) % 25 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self.criterion(self.model(s2_vl, a3_vl), a2_vl).item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"  Action Mapper training complete (best val loss: {best_val:.6f})")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"\nAction Mapper saved → {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"\nAction Mapper loaded ← {path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 1

    data_dir           = Path("./data/transfer_learning")
    traj_path          = data_dir / "trajectories.pkl"
    state_mapper_path  = data_dir / "state_mapper_3to2dof.pt"
    action_mapper_path = data_dir / "action_mapper_3to2dof.pt"

    print("\n" + "="*60)
    print("LOADING TRAJECTORIES")
    print("="*60)
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)
    meta = trajectories['metadata']
    print(f"  {meta['n_pairs']} pairs  |  source: {meta.get('source', '?')}"
          f"  |  seed: {meta['seed']}")
    print(f"  arm_obs 3dof : {meta.get('arm_size_3dof', '?')}D  |  "
          f"arm_obs 2dof : {meta.get('arm_size_2dof', '?')}D")
    if meta.get('source') != 'ppo_policy':
        print("\n  ⚠  WARNING: trajectories were NOT generated with PPO policies.")
        print("     Re-run eq_trajectories.py first for best results.\n")

    # ---- State Mapper ----
    print("\n" + "="*60)
    print("PART 1: STATE MAPPER  (2-dof arm_obs → 3-dof arm_obs)")
    print("="*60)
    st = StateMapperTrainer(device=device)
    st.train(trajectories, epochs=epochs, batch_size=512)
    st.save(str(state_mapper_path))

    # ---- Action Mapper ----
    print("\n" + "="*60)
    print("PART 2: ACTION MAPPER  (arm_obs_2dof + action_3dof → action_2dof)")
    print("="*60)
    at = ActionMapperTrainer(device=device)
    at.train(trajectories, epochs=epochs, batch_size=512)
    at.save(str(action_mapper_path))

    print("\n" + "="*60)
    print("MAPPER TRAINING COMPLETE")
    print("="*60)
    print(f"  State Mapper  → {state_mapper_path}")
    print(f"  Action Mapper → {action_mapper_path}")
    print("\nTransfer pipeline at inference (3-dof policy pilotant un bras 2-dof) :")
    print("  1. arm_obs_3dof_equiv = state_mapper(arm_obs_2dof)")
    print("     arm_obs_2dof  = full_obs_2dof[:8]")
    print("  2. full_obs_3dof_equiv = concat(arm_obs_3dof_equiv, task_obs_2dof)")
    print("     (ou utiliser directement arm_obs si la policy ne voit que le bras)")
    print("  3. obs_3dof_norm  = vec_norm_3dof.normalize_obs(full_obs_3dof_equiv)")
    print("  4. action_3dof    = policy_3dof(obs_3dof_norm)")
    print("  5. action_2dof    = action_mapper(arm_obs_2dof, action_3dof)   ← context-aware")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
