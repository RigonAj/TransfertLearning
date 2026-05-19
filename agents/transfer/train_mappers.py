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

from agents.transfer.mapper_models import StateMapperMLP, ActionMapperMLP

torch.set_num_threads(4)


def ee_from_state(state: np.ndarray, dof: int) -> np.ndarray:
    if dof == 2:
        theta1, theta2 = state[0], state[1]
        l1, l2 = 1.5, 1.5
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    else:
        theta1, theta2, theta3 = state[0], state[1], state[2]
        l1, l2, l3 = 1.0, 1.0, 1.0
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)
    return np.array([x, y], dtype=np.float32)


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


class Transfer2to3:
    def __init__(self, device: str = "cpu", log_dir: str = None):
        self.device = device
        self.writer = SummaryWriter(log_dir) if log_dir else None

        self.state_mapper = StateMapperMLP(8, 6).to(device)
        self.action_mapper = ActionMapperMLP(8, 2, 3).to(device)

        self.opt_state = optim.Adam(self.state_mapper.parameters(), lr=3e-4, weight_decay=1e-5)
        self.opt_action = optim.Adam(self.action_mapper.parameters(), lr=3e-4, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def train(self, trajectories: Dict, epochs: int = 1000, batch_size: int = 32, patience: int = 50):
        print("\n=== TRAINING TRANSFER 2→3 (on Reaching data with SEQUENCES) ===")

        segments_s3 = trajectories['segments_3dof']   # (N, L, 8)
        segments_s2 = trajectories['segments_2dof']   # (N, L, 6)
        segments_a2 = trajectories['segments_actions_2dof']
        segments_a3 = trajectories['segments_actions_3dof']

        N = segments_s3.shape[0]
        seq_len = segments_s3.shape[1]
        print(f"Segments: {N}, sequence length: {seq_len}")

        dataset = SegmentDataset(segments_s3, segments_s2, segments_a2, segments_a3)
        n_val = int(0.1 * N)
        n_train = N - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

        best_val = float('inf')
        wait = 0
        best_state = None
        best_action = None

        pbar = tqdm(range(epochs), desc="2→3 Transfer (seq)")
        for epoch in pbar:
            # --- Training ---
            self.state_mapper.train()
            self.action_mapper.train()
            train_loss_s = 0.0
            train_loss_a = 0.0

            for s3_seq, s2_seq, a2_seq, a3_seq in train_loader:
                # s3_seq: (B, L, 8)  -> on aplatit en (B*L, 8) pour appliquer le MLP
                B, L, D3 = s3_seq.shape
                s3_flat = s3_seq.view(B * L, D3)
                s2_flat = s2_seq.view(B * L, -1)
                a2_flat = a2_seq.view(B * L, -1)
                a3_flat = a3_seq.view(B * L, -1)

                # State mapper
                self.opt_state.zero_grad()
                s2_pred = self.state_mapper(s3_flat)
                loss_s = self.criterion(s2_pred, s2_flat)
                loss_s.backward()
                self.opt_state.step()
                train_loss_s += loss_s.item() * B  # pour pondérer par batch

                # Action mapper
                self.opt_action.zero_grad()
                a3_pred = self.action_mapper(s3_flat, a2_flat)
                loss_a = self.criterion(a3_pred, a3_flat)
                loss_a.backward()
                self.opt_action.step()
                train_loss_a += loss_a.item() * B

            train_loss_s /= len(train_dataset)
            train_loss_a /= len(train_dataset)

            # --- Validation ---
            self.state_mapper.eval()
            self.action_mapper.eval()
            val_loss_s = 0.0
            val_loss_a = 0.0
            with torch.no_grad():
                for s3_seq, s2_seq, a2_seq, a3_seq in val_loader:
                    B, L, D3 = s3_seq.shape
                    s3_flat = s3_seq.view(B * L, D3)
                    s2_flat = s2_seq.view(B * L, -1)
                    a2_flat = a2_seq.view(B * L, -1)
                    a3_flat = a3_seq.view(B * L, -1)

                    s2_pred = self.state_mapper(s3_flat)
                    loss_s = self.criterion(s2_pred, s2_flat)
                    a3_pred = self.action_mapper(s3_flat, a2_flat)
                    loss_a = self.criterion(a3_pred, a3_flat)

                    val_loss_s += loss_s.item() * B
                    val_loss_a += loss_a.item() * B

            val_loss_s /= len(val_dataset)
            val_loss_a /= len(val_dataset)
            total_val = val_loss_s + val_loss_a

            # Fidelity (sur un sous‑ensemble de segments entiers)
            fidelity = self._evaluate_fidelity_segments(segments_s3, segments_s2, segments_a2, segments_a3,
                                                        n_samples=min(200, N))

            if self.writer:
                self.writer.add_scalar("2to3/loss_state_val", val_loss_s, epoch)
                self.writer.add_scalar("2to3/loss_action_val", val_loss_a, epoch)
                self.writer.add_scalar("2to3/action_mse", fidelity['action_mse'], epoch)
                self.writer.add_scalar("2to3/ee_error", fidelity['ee_error'], epoch)

            pbar.set_postfix({
                "s_val": f"{val_loss_s:.5f}",
                "a_val": f"{val_loss_a:.5f}",
                "a_mse": f"{fidelity['action_mse']:.5f}",
                "ee_err": f"{fidelity['ee_error']:.5f}"
            })

            if total_val < best_val - 1e-5:
                best_val = total_val
                wait = 0
                best_state = {k: v.clone() for k, v in self.state_mapper.state_dict().items()}
                best_action = {k: v.clone() for k, v in self.action_mapper.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping")
                    break

        if best_state:
            self.state_mapper.load_state_dict(best_state)
            self.action_mapper.load_state_dict(best_action)

        print(f"2→3 Training done - Best val loss: {best_val:.6f}")

    @torch.no_grad()
    def _evaluate_fidelity_segments(self, segments_s3, segments_s2, segments_a2, segments_a3, n_samples=200):
        """Évalue la fidélité sur des segments complets (non aplatis)."""
        idx = np.random.choice(len(segments_s3), n_samples, replace=False)
        total_action_mse = 0.0
        total_ee_error = 0.0

        for i in idx:
            s3_seq = torch.from_numpy(segments_s3[i]).float().to(self.device)  # (L, 8)
            s2_seq = torch.from_numpy(segments_s2[i]).float().to(self.device)  # (L, 6)
            a2_seq = torch.from_numpy(segments_a2[i]).float().to(self.device)  # (L, 2)
            a3_seq = torch.from_numpy(segments_a3[i]).float().to(self.device)  # (L, 3)

            L = s3_seq.shape[0]
            s2_pred = self.state_mapper(s3_seq)          # (L, 6)
            a3_pred = self.action_mapper(s3_seq, a2_seq) # (L, 3)

            action_mse = self.criterion(a3_pred, a3_seq).item()
            total_action_mse += action_mse

            # Erreur d'effecteur sur toute la séquence (moyenne)
            ee_err_seq = 0.0
            for t in range(L):
                ee_pred = ee_from_state(s2_pred[t].cpu().numpy(), 2)
                ee_real = ee_from_state(s3_seq[t].cpu().numpy(), 3)
                ee_err_seq += np.linalg.norm(ee_pred - ee_real) ** 2
            total_ee_error += ee_err_seq / L

        return {
            'action_mse': total_action_mse / n_samples,
            'ee_error': total_ee_error / n_samples
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_mapper': self.state_mapper.state_dict(),
            'action_mapper': self.action_mapper.state_dict(),
        }, path)
        print(f"✓ Saved 2→3 → {path}")


class Transfer3to2:
    def __init__(self, device: str = "cpu", log_dir: str = None):
        self.device = device
        self.writer = SummaryWriter(log_dir) if log_dir else None

        self.state_mapper = StateMapperMLP(6, 8).to(device)
        self.action_mapper = ActionMapperMLP(6, 3, 2).to(device)

        self.opt_state = optim.Adam(self.state_mapper.parameters(), lr=3e-4, weight_decay=1e-5)
        self.opt_action = optim.Adam(self.action_mapper.parameters(), lr=3e-4, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def train(self, trajectories: Dict, epochs: int = 1000, batch_size: int = 32, patience: int = 60):
        print("\n=== TRAINING TRANSFER 3→2 (on Reaching data with SEQUENCES) ===")

        segments_s3 = trajectories['segments_3dof']
        segments_s2 = trajectories['segments_2dof']
        segments_a2 = trajectories['segments_actions_2dof']
        segments_a3 = trajectories['segments_actions_3dof']

        N = segments_s3.shape[0]
        seq_len = segments_s3.shape[1]
        print(f"Segments: {N}, sequence length: {seq_len}")

        dataset = SegmentDataset(segments_s3, segments_s2, segments_a2, segments_a3)  # mêmes données, ordre différent pour 3→2
        # Note: pour 3→2 on utilise state_mapper(s2) -> s3, action_mapper(s2, a3) -> a2
        # On peut réutiliser le même dataset, mais il faut l'adapter. Je crée un dataset spécifique:
        class SegDataset3to2(Dataset):
            def __init__(self, s2, s3, a3, a2):
                self.s2 = torch.from_numpy(s2).float()
                self.s3 = torch.from_numpy(s3).float()
                self.a3 = torch.from_numpy(a3).float()
                self.a2 = torch.from_numpy(a2).float()
            def __len__(self): return len(self.s2)
            def __getitem__(self, i): return self.s2[i], self.s3[i], self.a3[i], self.a2[i]

        dataset_32 = SegDataset3to2(segments_s2, segments_s3, segments_a3, segments_a2)
        n_val = int(0.1 * N)
        n_train = N - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(dataset_32, [n_train, n_val])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

        best_val = float('inf')
        wait = 0
        best_state = None
        best_action = None

        pbar = tqdm(range(epochs), desc="3→2 Transfer (seq)")
        for epoch in pbar:
            self.state_mapper.train()
            self.action_mapper.train()
            train_loss_s = 0.0
            train_loss_a = 0.0

            for s2_seq, s3_seq, a3_seq, a2_seq in train_loader:
                B, L, D2 = s2_seq.shape
                s2_flat = s2_seq.view(B * L, D2)
                s3_flat = s3_seq.view(B * L, -1)
                a3_flat = a3_seq.view(B * L, -1)
                a2_flat = a2_seq.view(B * L, -1)

                self.opt_state.zero_grad()
                s3_pred = self.state_mapper(s2_flat)
                loss_s = self.criterion(s3_pred, s3_flat)
                loss_s.backward()
                self.opt_state.step()
                train_loss_s += loss_s.item() * B

                self.opt_action.zero_grad()
                a2_pred = self.action_mapper(s2_flat, a3_flat)
                loss_a = self.criterion(a2_pred, a2_flat)
                loss_a.backward()
                self.opt_action.step()
                train_loss_a += loss_a.item() * B

            train_loss_s /= len(train_dataset)
            train_loss_a /= len(train_dataset)

            self.state_mapper.eval()
            self.action_mapper.eval()
            val_loss_s = 0.0
            val_loss_a = 0.0
            with torch.no_grad():
                for s2_seq, s3_seq, a3_seq, a2_seq in val_loader:
                    B, L, D2 = s2_seq.shape
                    s2_flat = s2_seq.view(B * L, D2)
                    s3_flat = s3_seq.view(B * L, -1)
                    a3_flat = a3_seq.view(B * L, -1)
                    a2_flat = a2_seq.view(B * L, -1)

                    s3_pred = self.state_mapper(s2_flat)
                    loss_s = self.criterion(s3_pred, s3_flat)
                    a2_pred = self.action_mapper(s2_flat, a3_flat)
                    loss_a = self.criterion(a2_pred, a2_flat)

                    val_loss_s += loss_s.item() * B
                    val_loss_a += loss_a.item() * B

            val_loss_s /= len(val_dataset)
            val_loss_a /= len(val_dataset)
            total_val = val_loss_s + val_loss_a

            # Fidelity
            fidelity = self._evaluate_fidelity_segments(segments_s2, segments_s3, segments_a3, segments_a2,
                                                        n_samples=min(200, N))

            if self.writer:
                self.writer.add_scalar("3to2/loss_state_val", val_loss_s, epoch)
                self.writer.add_scalar("3to2/loss_action_val", val_loss_a, epoch)
                self.writer.add_scalar("3to2/action_mse", fidelity['action_mse'], epoch)
                self.writer.add_scalar("3to2/ee_error", fidelity['ee_error'], epoch)

            pbar.set_postfix({
                "s_val": f"{val_loss_s:.5f}",
                "a_val": f"{val_loss_a:.5f}",
                "a_mse": f"{fidelity['action_mse']:.5f}",
                "ee_err": f"{fidelity['ee_error']:.5f}"
            })

            if total_val < best_val - 1e-5:
                best_val = total_val
                wait = 0
                best_state = {k: v.clone() for k, v in self.state_mapper.state_dict().items()}
                best_action = {k: v.clone() for k, v in self.action_mapper.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping")
                    break

        if best_state:
            self.state_mapper.load_state_dict(best_state)
            self.action_mapper.load_state_dict(best_action)

        print(f"3→2 Training done - Best val loss: {best_val:.6f}")

    @torch.no_grad()
    def _evaluate_fidelity_segments(self, segments_s2, segments_s3, segments_a3, segments_a2, n_samples=200):
        idx = np.random.choice(len(segments_s2), n_samples, replace=False)
        total_action_mse = 0.0
        total_ee_error = 0.0

        for i in idx:
            s2_seq = torch.from_numpy(segments_s2[i]).float().to(self.device)  # (L, 6)
            s3_seq = torch.from_numpy(segments_s3[i]).float().to(self.device)  # (L, 8)
            a3_seq = torch.from_numpy(segments_a3[i]).float().to(self.device)  # (L, 3)
            a2_seq = torch.from_numpy(segments_a2[i]).float().to(self.device)  # (L, 2)

            L = s2_seq.shape[0]
            s3_pred = self.state_mapper(s2_seq)
            a2_pred = self.action_mapper(s2_seq, a3_seq)

            action_mse = self.criterion(a2_pred, a2_seq).item()
            total_action_mse += action_mse

            ee_err_seq = 0.0
            for t in range(L):
                ee_pred = ee_from_state(s3_pred[t].cpu().numpy(), 3)
                ee_real = ee_from_state(s2_seq[t].cpu().numpy(), 2)
                ee_err_seq += np.linalg.norm(ee_pred - ee_real) ** 2
            total_ee_error += ee_err_seq / L

        return {
            'action_mse': total_action_mse / n_samples,
            'ee_error': total_ee_error / n_samples
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_mapper': self.state_mapper.state_dict(),
            'action_mapper': self.action_mapper.state_dict(),
        }, path)
        print(f"✓ Saved 3→2 → {path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("./data/DIRECT")

    traj_path = data_dir / "trajectories_aligned_ss.pkl"   # nouveau fichier
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)

    print(f"Loaded {trajectories['metadata']['n_segments']} segments of length {trajectories['metadata']['seq_len']}")

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
