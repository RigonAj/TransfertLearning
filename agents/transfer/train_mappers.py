import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict

# FIX #1 : chemin d'import corrigé (était agents.transfer.mapper_models)
from .mapper_models import StateMapperMLP, ActionMapperMLP

torch.set_num_threads(1)


# =========================================================
# Utilitaire : position cartésienne depuis un état normalisé
# =========================================================

def ee_from_state(state: np.ndarray, dof: int) -> np.ndarray:
    """
    Calcule la position de l'effecteur à partir d'un état normalisé.

    FIX #2 : les états stockés ont θ/π (normalisé).  Il faut donc
    dénormaliser par π avant de passer à cos/sin.
    Géométrie :
      - 2DoF : l1=l2=1.5  (max_reach=3.0)
      - 3DoF : l1=l2=l3=1.0 (max_reach=3.0)
    """
    if dof == 2:
        theta1 = state[0] * np.pi      # dénormalisation
        theta2 = state[1] * np.pi
        l1, l2 = 1.5, 1.5
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    else:  # dof == 3
        theta1 = state[0] * np.pi
        theta2 = state[1] * np.pi
        theta3 = state[2] * np.pi
        l1, l2, l3 = 1.0, 1.0, 1.0
        x = (l1 * np.cos(theta1)
             + l2 * np.cos(theta1 + theta2)
             + l3 * np.cos(theta1 + theta2 + theta3))
        y = (l1 * np.sin(theta1)
             + l2 * np.sin(theta1 + theta2)
             + l3 * np.sin(theta1 + theta2 + theta3))
    return np.array([x, y], dtype=np.float32)


# =========================================================
# Dataset commun
# =========================================================

class SegmentDataset(Dataset):
    """Dataset qui retourne un segment complet (séquence) à la fois."""
    def __init__(self, segments_s3, segments_s2, segments_a2, segments_a3):
        self.s3 = torch.from_numpy(segments_s3).float()  # (N, L, 8)
        self.s2 = torch.from_numpy(segments_s2).float()  # (N, L, 6)
        self.a2 = torch.from_numpy(segments_a2).float()  # (N, L, 2)
        self.a3 = torch.from_numpy(segments_a3).float()  # (N, L, 3)

    def __len__(self):
        return len(self.s3)

    def __getitem__(self, idx):
        return self.s3[idx], self.s2[idx], self.a2[idx], self.a3[idx]


# =========================================================
# Transfer 2DoF → 3DoF
# =========================================================

class Transfer2to3:
    def __init__(self, device: str = "cpu", log_dir: str = None):
        self.device  = device
        self.writer  = SummaryWriter(log_dir) if log_dir else None

        # s3 (8D) → s2 (6D)  |  (s3, a2) → a3
        self.state_mapper  = StateMapperMLP(8, 6).to(device)
        self.action_mapper = ActionMapperMLP(8, 2, 3).to(device)

        self.opt_state  = optim.Adam(self.state_mapper.parameters(),  lr=3e-4, weight_decay=1e-5)
        self.opt_action = optim.Adam(self.action_mapper.parameters(), lr=3e-4, weight_decay=1e-5)
        self.criterion  = nn.MSELoss()

    # ------------------------------------------------------------------
    def train(self, trajectories: Dict, epochs: int = 1000,
              batch_size: int = 32, patience: int = 50):
        print("\n=== TRAINING TRANSFER 2→3 (sequences, spatial-sampling aligné) ===")

        segments_s3 = trajectories['segments_3dof']          # (N, L, 8)
        segments_s2 = trajectories['segments_2dof']          # (N, L, 6)
        segments_a2 = trajectories['segments_actions_2dof']  # (N, L, 2)
        segments_a3 = trajectories['segments_actions_3dof']  # (N, L, 3)

        N      = segments_s3.shape[0]
        seq_len = segments_s3.shape[1]
        print(f"Segments: {N}, longueur séquence: {seq_len}")

        # FIX #5 : split manuel pour pouvoir passer les indices val à fidelity
        rng_split = np.random.RandomState(0)
        idx_all   = rng_split.permutation(N)
        n_val     = max(1, int(0.1 * N))
        n_train   = N - n_val
        val_idx   = idx_all[:n_val]
        train_idx = idx_all[n_val:]

        dataset = SegmentDataset(segments_s3, segments_s2, segments_a2, segments_a3)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset   = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True,  pin_memory=False)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                                  shuffle=False, pin_memory=False)

        best_val    = float('inf')
        wait        = 0
        best_state  = None
        best_action = None

        pbar = tqdm(range(epochs), desc="2→3 Transfer")
        for epoch in pbar:
            # --- Entraînement ---
            self.state_mapper.train()
            self.action_mapper.train()
            train_loss_s = 0.0
            train_loss_a = 0.0

            for s3_seq, s2_seq, a2_seq, a3_seq in train_loader:
                B, L, D3 = s3_seq.shape
                s3_flat  = s3_seq.view(B * L, D3).to(self.device)
                s2_flat  = s2_seq.view(B * L, -1).to(self.device)
                a2_flat  = a2_seq.view(B * L, -1).to(self.device)
                a3_flat  = a3_seq.view(B * L, -1).to(self.device)

                self.opt_state.zero_grad()
                s2_pred = self.state_mapper(s3_flat)
                loss_s  = self.criterion(s2_pred, s2_flat)
                loss_s.backward()
                self.opt_state.step()

                self.opt_action.zero_grad()
                a3_pred = self.action_mapper(s3_flat, a2_flat)
                loss_a  = self.criterion(a3_pred, a3_flat)
                loss_a.backward()
                self.opt_action.step()

                # FIX #6 : MSELoss fait déjà la moyenne sur le batch ;
                # on accumule et on divise par le nombre de batches.
                train_loss_s += loss_s.item()
                train_loss_a += loss_a.item()

            train_loss_s /= len(train_loader)
            train_loss_a /= len(train_loader)

            # --- Validation ---
            self.state_mapper.eval()
            self.action_mapper.eval()
            val_loss_s = 0.0
            val_loss_a = 0.0
            with torch.no_grad():
                for s3_seq, s2_seq, a2_seq, a3_seq in val_loader:
                    B, L, D3 = s3_seq.shape
                    s3_flat = s3_seq.view(B * L, D3).to(self.device)
                    s2_flat = s2_seq.view(B * L, -1).to(self.device)
                    a2_flat = a2_seq.view(B * L, -1).to(self.device)
                    a3_flat = a3_seq.view(B * L, -1).to(self.device)

                    val_loss_s += self.criterion(self.state_mapper(s3_flat),          s2_flat).item()
                    val_loss_a += self.criterion(self.action_mapper(s3_flat, a2_flat), a3_flat).item()

            val_loss_s /= len(val_loader)
            val_loss_a /= len(val_loader)
            total_val   = val_loss_s + val_loss_a

            # FIX #5 : fidelity calculée sur le VAL set uniquement
            fidelity = self._evaluate_fidelity_segments(
                segments_s3[val_idx], segments_s2[val_idx],
                segments_a2[val_idx], segments_a3[val_idx],
                n_samples=min(200, n_val),
            )

            if self.writer:
                self.writer.add_scalar("2to3/train_loss_state",  train_loss_s,            epoch)
                self.writer.add_scalar("2to3/train_loss_action", train_loss_a,            epoch)
                self.writer.add_scalar("2to3/val_loss_state",    val_loss_s,              epoch)
                self.writer.add_scalar("2to3/val_loss_action",   val_loss_a,              epoch)
                self.writer.add_scalar("2to3/action_mse",        fidelity['action_mse'],  epoch)
                self.writer.add_scalar("2to3/ee_error",          fidelity['ee_error'],    epoch)

            pbar.set_postfix({
                "s_val":  f"{val_loss_s:.5f}",
                "a_val":  f"{val_loss_a:.5f}",
                "a_mse":  f"{fidelity['action_mse']:.5f}",
                "ee_err": f"{fidelity['ee_error']:.5f}",
            })

            if total_val < best_val - 1e-5:
                best_val    = total_val
                wait        = 0
                best_state  = {k: v.clone() for k, v in self.state_mapper.state_dict().items()}
                best_action = {k: v.clone() for k, v in self.action_mapper.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping")
                    break

        if best_state:
            self.state_mapper.load_state_dict(best_state)
            self.action_mapper.load_state_dict(best_action)

        print(f"2→3 Training done  |  Best val loss: {best_val:.6f}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_fidelity_segments(self, segments_s3, segments_s2,
                                     segments_a2, segments_a3, n_samples=200):
        """Fidélité sur des segments complets (non aplatis)."""
        n = min(n_samples, len(segments_s3))
        idx = np.random.choice(len(segments_s3), n, replace=False)
        total_action_mse = 0.0
        total_ee_error   = 0.0

        self.state_mapper.eval()
        self.action_mapper.eval()

        for i in idx:
            s3_seq = torch.from_numpy(segments_s3[i]).float().to(self.device)  # (L, 8)
            s2_seq = torch.from_numpy(segments_s2[i]).float().to(self.device)  # (L, 6)
            a2_seq = torch.from_numpy(segments_a2[i]).float().to(self.device)  # (L, 2)
            a3_seq = torch.from_numpy(segments_a3[i]).float().to(self.device)  # (L, 3)

            s2_pred = self.state_mapper(s3_seq)           # (L, 6)
            a3_pred = self.action_mapper(s3_seq, a2_seq)  # (L, 3)

            total_action_mse += self.criterion(a3_pred, a3_seq).item()

            L = s3_seq.shape[0]
            ee_err_seq = 0.0
            for t in range(L):
                # FIX #2 : ee_from_state dénormalise θ×π correctement
                ee_pred = ee_from_state(s2_pred[t].cpu().numpy(), dof=2)
                ee_real = ee_from_state(s3_seq[t].cpu().numpy(),  dof=3)
                ee_err_seq += np.linalg.norm(ee_pred - ee_real) ** 2
            total_ee_error += ee_err_seq / L

        return {
            'action_mse': total_action_mse / n,
            'ee_error':   total_ee_error   / n,
        }

    # ------------------------------------------------------------------
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_mapper':  self.state_mapper.state_dict(),
            'action_mapper': self.action_mapper.state_dict(),
        }, path)
        print(f"✓ Saved 2→3 → {path}")


# =========================================================
# Transfer 3DoF → 2DoF
# =========================================================

class Transfer3to2:
    def __init__(self, device: str = "cpu", log_dir: str = None):
        self.device  = device
        self.writer  = SummaryWriter(log_dir) if log_dir else None

        # s2 (6D) → s3 (8D)  |  (s2, a3) → a2
        self.state_mapper  = StateMapperMLP(6, 8).to(device)
        self.action_mapper = ActionMapperMLP(6, 3, 2).to(device)

        self.opt_state  = optim.Adam(self.state_mapper.parameters(),  lr=3e-4, weight_decay=1e-5)
        self.opt_action = optim.Adam(self.action_mapper.parameters(), lr=3e-4, weight_decay=1e-5)
        self.criterion  = nn.MSELoss()

    # ------------------------------------------------------------------
    def train(self, trajectories: Dict, epochs: int = 1000,
              batch_size: int = 32, patience: int = 60):
        print("\n=== TRAINING TRANSFER 3→2 (sequences, spatial-sampling aligné) ===")

        segments_s3 = trajectories['segments_3dof']          # (N, L, 8)
        segments_s2 = trajectories['segments_2dof']          # (N, L, 6)
        segments_a2 = trajectories['segments_actions_2dof']  # (N, L, 2)
        segments_a3 = trajectories['segments_actions_3dof']  # (N, L, 3)

        N      = segments_s2.shape[0]
        seq_len = segments_s2.shape[1]
        print(f"Segments: {N}, longueur séquence: {seq_len}")

        # FIX #5 : split manuel
        rng_split = np.random.RandomState(0)
        idx_all   = rng_split.permutation(N)
        n_val     = max(1, int(0.1 * N))
        n_train   = N - n_val
        val_idx   = idx_all[:n_val]
        train_idx = idx_all[n_val:]

        class SegDataset3to2(Dataset):
            def __init__(self, s2, s3, a3, a2):
                self.s2 = torch.from_numpy(s2).float()
                self.s3 = torch.from_numpy(s3).float()
                self.a3 = torch.from_numpy(a3).float()
                self.a2 = torch.from_numpy(a2).float()
            def __len__(self): return len(self.s2)
            def __getitem__(self, i): return self.s2[i], self.s3[i], self.a3[i], self.a2[i]

        dataset      = SegDataset3to2(segments_s2, segments_s3, segments_a3, segments_a2)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset   = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True,  pin_memory=False)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                                  shuffle=False, pin_memory=False)

        best_val    = float('inf')
        wait        = 0
        best_state  = None
        best_action = None

        pbar = tqdm(range(epochs), desc="3→2 Transfer")
        for epoch in pbar:
            self.state_mapper.train()
            self.action_mapper.train()
            train_loss_s = 0.0
            train_loss_a = 0.0

            for s2_seq, s3_seq, a3_seq, a2_seq in train_loader:
                B, L, D2 = s2_seq.shape
                s2_flat = s2_seq.view(B * L, D2).to(self.device)
                s3_flat = s3_seq.view(B * L, -1).to(self.device)
                a3_flat = a3_seq.view(B * L, -1).to(self.device)
                a2_flat = a2_seq.view(B * L, -1).to(self.device)

                self.opt_state.zero_grad()
                s3_pred = self.state_mapper(s2_flat)
                loss_s  = self.criterion(s3_pred, s3_flat)
                loss_s.backward()
                self.opt_state.step()

                self.opt_action.zero_grad()
                a2_pred = self.action_mapper(s2_flat, a3_flat)
                loss_a  = self.criterion(a2_pred, a2_flat)
                loss_a.backward()
                self.opt_action.step()

                # FIX #6
                train_loss_s += loss_s.item()
                train_loss_a += loss_a.item()

            train_loss_s /= len(train_loader)
            train_loss_a /= len(train_loader)

            self.state_mapper.eval()
            self.action_mapper.eval()
            val_loss_s = 0.0
            val_loss_a = 0.0
            with torch.no_grad():
                for s2_seq, s3_seq, a3_seq, a2_seq in val_loader:
                    B, L, D2 = s2_seq.shape
                    s2_flat = s2_seq.view(B * L, D2).to(self.device)
                    s3_flat = s3_seq.view(B * L, -1).to(self.device)
                    a3_flat = a3_seq.view(B * L, -1).to(self.device)
                    a2_flat = a2_seq.view(B * L, -1).to(self.device)

                    val_loss_s += self.criterion(self.state_mapper(s2_flat),          s3_flat).item()
                    val_loss_a += self.criterion(self.action_mapper(s2_flat, a3_flat), a2_flat).item()

            val_loss_s /= len(val_loader)
            val_loss_a /= len(val_loader)
            total_val   = val_loss_s + val_loss_a

            # FIX #5 : fidelity sur le VAL set uniquement
            fidelity = self._evaluate_fidelity_segments(
                segments_s2[val_idx], segments_s3[val_idx],
                segments_a3[val_idx], segments_a2[val_idx],
                n_samples=min(200, n_val),
            )

            if self.writer:
                self.writer.add_scalar("3to2/train_loss_state",  train_loss_s,            epoch)
                self.writer.add_scalar("3to2/train_loss_action", train_loss_a,            epoch)
                self.writer.add_scalar("3to2/val_loss_state",    val_loss_s,              epoch)
                self.writer.add_scalar("3to2/val_loss_action",   val_loss_a,              epoch)
                self.writer.add_scalar("3to2/action_mse",        fidelity['action_mse'],  epoch)
                self.writer.add_scalar("3to2/ee_error",          fidelity['ee_error'],    epoch)

            pbar.set_postfix({
                "s_val":  f"{val_loss_s:.5f}",
                "a_val":  f"{val_loss_a:.5f}",
                "a_mse":  f"{fidelity['action_mse']:.5f}",
                "ee_err": f"{fidelity['ee_error']:.5f}",
            })

            if total_val < best_val - 1e-5:
                best_val    = total_val
                wait        = 0
                best_state  = {k: v.clone() for k, v in self.state_mapper.state_dict().items()}
                best_action = {k: v.clone() for k, v in self.action_mapper.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping")
                    break

        if best_state:
            self.state_mapper.load_state_dict(best_state)
            self.action_mapper.load_state_dict(best_action)

        print(f"3→2 Training done  |  Best val loss: {best_val:.6f}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_fidelity_segments(self, segments_s2, segments_s3,
                                     segments_a3, segments_a2, n_samples=200):
        """Fidélité sur des segments complets (non aplatis)."""
        n = min(n_samples, len(segments_s2))
        idx = np.random.choice(len(segments_s2), n, replace=False)
        total_action_mse = 0.0
        total_ee_error   = 0.0

        self.state_mapper.eval()
        self.action_mapper.eval()

        for i in idx:
            s2_seq = torch.from_numpy(segments_s2[i]).float().to(self.device)  # (L, 6)
            s3_seq = torch.from_numpy(segments_s3[i]).float().to(self.device)  # (L, 8)
            a3_seq = torch.from_numpy(segments_a3[i]).float().to(self.device)  # (L, 3)
            a2_seq = torch.from_numpy(segments_a2[i]).float().to(self.device)  # (L, 2)

            s3_pred = self.state_mapper(s2_seq)           # (L, 8)
            a2_pred = self.action_mapper(s2_seq, a3_seq)  # (L, 2)

            total_action_mse += self.criterion(a2_pred, a2_seq).item()

            L = s2_seq.shape[0]
            ee_err_seq = 0.0
            for t in range(L):
                # FIX #2 : dénormalisation correcte
                ee_pred = ee_from_state(s3_pred[t].cpu().numpy(), dof=3)
                ee_real = ee_from_state(s2_seq[t].cpu().numpy(),  dof=2)
                ee_err_seq += np.linalg.norm(ee_pred - ee_real) ** 2
            total_ee_error += ee_err_seq / L

        return {
            'action_mse': total_action_mse / n,
            'ee_error':   total_ee_error   / n,
        }

    # ------------------------------------------------------------------
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_mapper':  self.state_mapper.state_dict(),
            'action_mapper': self.action_mapper.state_dict(),
        }, path)
        print(f"✓ Saved 3→2 → {path}")


# =========================================================
# Point d'entrée
# =========================================================

def main():
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("./data/DIRECT")

    traj_path = data_dir / "trajectories_aligned_ss.pkl"
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)

    meta = trajectories['metadata']
    print(f"Loaded {meta['n_segments']} segments of length {meta['seq_len']}")

    runs_dir = Path("./data/DIRECT/mappers_logs")
    runs_dir.mkdir(exist_ok=True)

    t23 = Transfer2to3(device=device, log_dir=str(runs_dir / "transfer_2to3_seq"))
    t23.train(trajectories, epochs=1200, batch_size=32, patience=50)
    t23.save(data_dir / "transfer_2to3_seq.pt")

    t32 = Transfer3to2(device=device, log_dir=str(runs_dir / "transfer_3to2_seq"))
    t32.train(trajectories, epochs=1200, batch_size=32, patience=60)
    t32.save(data_dir / "transfer_3to2_seq.pt")

    print("\n=== ENTRAÎNEMENT TERMINÉ ===")
    print("→ Lance tensorboard avec : tensorboard --logdir ./data")


if __name__ == "__main__":
    main()
