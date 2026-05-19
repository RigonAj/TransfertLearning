import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict

from agents.transfer.mapper_models import StateMapperMLP, ActionMapperMLP

torch.set_num_threads(2)


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


class Transfer2to3:
    def __init__(self, device: str = "cpu", log_dir: str = None):
        self.device = device
        self.writer = SummaryWriter(log_dir) if log_dir else None

        self.state_mapper = StateMapperMLP(8, 6).to(device)
        self.action_mapper = ActionMapperMLP(8, 2, 3).to(device)

        self.opt_state = optim.Adam(self.state_mapper.parameters(), lr=3e-4, weight_decay=1e-5)
        self.opt_action = optim.Adam(self.action_mapper.parameters(), lr=3e-4, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def train(self, trajectories: Dict, epochs: int = 1000, batch_size: int = 512, patience: int = 50):
        print("\n=== TRAINING TRANSFER 2→3 (on Reaching data) ===")

        # ALIGNEMENT DES DONNÉES
        min_len = min(
            len(trajectories['states_3dof']),
            len(trajectories['states_2dof']),
            len(trajectories['actions_2dof']),
            len(trajectories['actions_3dof'])
        )
        print(f"Aligning data to minimum length: {min_len} samples")

        s3 = torch.tensor(trajectories['states_3dof'][:min_len], dtype=torch.float32).to(self.device)
        s2 = torch.tensor(trajectories['states_2dof'][:min_len], dtype=torch.float32).to(self.device)
        a2 = torch.tensor(trajectories['actions_2dof'][:min_len], dtype=torch.float32).to(self.device)
        a3 = torch.tensor(trajectories['actions_3dof'][:min_len], dtype=torch.float32).to(self.device)

        n = len(s3)
        idx = torch.randperm(n)
        cut = int(n * 0.9)
        train_idx, val_idx = idx[:cut], idx[cut:]

        train_loader = DataLoader(TensorDataset(s3[train_idx], s2[train_idx], a2[train_idx], a3[train_idx]),
                                  batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(TensorDataset(s3[val_idx], s2[val_idx], a2[val_idx], a3[val_idx]),
                                batch_size=batch_size*2, shuffle=False, pin_memory=True)

        best_val = float('inf')
        wait = 0
        best_state = None
        best_action = None

        pbar = tqdm(range(epochs), desc="2→3 Transfer")
        for epoch in pbar:
            # Training
            self.state_mapper.train()
            self.action_mapper.train()
            train_loss_s = train_loss_a = 0.0

            for s3b, s2b, a2b, a3b in train_loader:
                self.opt_state.zero_grad()
                loss_s = self.criterion(self.state_mapper(s3b), s2b)
                loss_s.backward()
                self.opt_state.step()
                train_loss_s += loss_s.item()

                self.opt_action.zero_grad()
                pred = self.action_mapper(s3b, a2b)
                loss_a = self.criterion(pred, a3b)
                loss_a.backward()
                self.opt_action.step()
                train_loss_a += loss_a.item()

            train_loss_s /= len(train_loader)
            train_loss_a /= len(train_loader)

            # Validation
            self.state_mapper.eval()
            self.action_mapper.eval()
            val_loss_s = val_loss_a = 0.0
            with torch.no_grad():
                for s3b, s2b, a2b, a3b in val_loader:
                    val_loss_s += self.criterion(self.state_mapper(s3b), s2b).item()
                    val_loss_a += self.criterion(self.action_mapper(s3b, a2b), a3b).item()

            val_loss_s /= len(val_loader)
            val_loss_a /= len(val_loader)
            total_val = val_loss_s + val_loss_a

            # Fidelity (améliorée)
            fidelity = self._evaluate_fidelity(trajectories, min_len)

            if self.writer:
                self.writer.add_scalar("2to3/loss_state_val", val_loss_s, epoch)
                self.writer.add_scalar("2to3/loss_action_val", val_loss_a, epoch)
                self.writer.add_scalar("2to3/action_mse", fidelity['action_mse'], epoch)
                self.writer.add_scalar("2to3/ee_error", fidelity['ee_error'], epoch)
                self.writer.add_scalar("2to3/total_fidelity", fidelity['action_mse'] + 10 * fidelity['ee_error'], epoch)

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
    def _evaluate_fidelity(self, trajectories: Dict, min_len: int, n_samples=2048):
        n_samples = min(n_samples, min_len)
        idx = np.random.choice(min_len, n_samples, replace=False)

        s3 = torch.tensor(trajectories['states_3dof'][idx], device=self.device)
        a2 = torch.tensor(trajectories['actions_2dof'][idx], device=self.device)
        a3 = torch.tensor(trajectories['actions_3dof'][idx], device=self.device)

        s2_pred = self.state_mapper(s3)
        a3_pred = self.action_mapper(s3, a2)

        action_mse = self.criterion(a3_pred, a3).item()

        # Erreur d'effecteur terminal (end‑effector)
        ee_error = 0.0
        n_eval = min(512, len(s3))
        for i in range(n_eval):
            ee_pred = ee_from_state(s2_pred[i].cpu().numpy(), 2)
            ee_real = ee_from_state(s3[i].cpu().numpy(), 3)
            ee_error += np.linalg.norm(ee_pred - ee_real) ** 2
        ee_error /= n_eval

        return {
            'action_mse': action_mse,
            'ee_error': float(ee_error)
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

    def train(self, trajectories: Dict, epochs: int = 1000, batch_size: int = 512, patience: int = 60):
        print("\n=== TRAINING TRANSFER 3→2 (on Reaching data) ===")

        min_len = min(
            len(trajectories['states_2dof']),
            len(trajectories['states_3dof']),
            len(trajectories['actions_3dof']),
            len(trajectories['actions_2dof'])
        )
        print(f"Aligning data to minimum length: {min_len} samples")

        s2 = torch.tensor(trajectories['states_2dof'][:min_len], dtype=torch.float32).to(self.device)
        s3 = torch.tensor(trajectories['states_3dof'][:min_len], dtype=torch.float32).to(self.device)
        a3 = torch.tensor(trajectories['actions_3dof'][:min_len], dtype=torch.float32).to(self.device)
        a2 = torch.tensor(trajectories['actions_2dof'][:min_len], dtype=torch.float32).to(self.device)

        n = len(s2)
        idx = torch.randperm(n)
        cut = int(n * 0.9)
        train_idx, val_idx = idx[:cut], idx[cut:]

        train_loader = DataLoader(TensorDataset(s2[train_idx], s3[train_idx], a3[train_idx], a2[train_idx]),
                                  batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(TensorDataset(s2[val_idx], s3[val_idx], a3[val_idx], a2[val_idx]),
                                batch_size=batch_size*2, shuffle=False, pin_memory=True)

        best_val = float('inf')
        wait = 0
        best_state = None
        best_action = None

        pbar = tqdm(range(epochs), desc="3→2 Transfer")
        for epoch in pbar:
            # Training
            self.state_mapper.train()
            self.action_mapper.train()
            train_loss_s = train_loss_a = 0.0

            for s2b, s3b, a3b, a2b in train_loader:
                self.opt_state.zero_grad()
                loss_s = self.criterion(self.state_mapper(s2b), s3b)
                loss_s.backward()
                self.opt_state.step()
                train_loss_s += loss_s.item()

                self.opt_action.zero_grad()
                pred = self.action_mapper(s2b, a3b)
                loss_a = self.criterion(pred, a2b)
                loss_a.backward()
                self.opt_action.step()
                train_loss_a += loss_a.item()

            train_loss_s /= len(train_loader)
            train_loss_a /= len(train_loader)

            # Validation
            self.state_mapper.eval()
            self.action_mapper.eval()
            val_loss_s = val_loss_a = 0.0
            with torch.no_grad():
                for s2b, s3b, a3b, a2b in val_loader:
                    val_loss_s += self.criterion(self.state_mapper(s2b), s3b).item()
                    val_loss_a += self.criterion(self.action_mapper(s2b, a3b), a2b).item()

            val_loss_s /= len(val_loader)
            val_loss_a /= len(val_loader)
            total_val = val_loss_s + val_loss_a

            # Fidelity (améliorée)
            fidelity = self._evaluate_fidelity(trajectories, min_len)

            if self.writer:
                self.writer.add_scalar("3to2/loss_state_val", val_loss_s, epoch)
                self.writer.add_scalar("3to2/loss_action_val", val_loss_a, epoch)
                self.writer.add_scalar("3to2/action_mse", fidelity['action_mse'], epoch)
                self.writer.add_scalar("3to2/ee_error", fidelity['ee_error'], epoch)
                self.writer.add_scalar("3to2/total_fidelity", fidelity['action_mse'] + 10 * fidelity['ee_error'], epoch)

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
    def _evaluate_fidelity(self, trajectories: Dict, min_len: int, n_samples=2048):
        n_samples = min(n_samples, min_len)
        idx = np.random.choice(min_len, n_samples, replace=False)

        s2 = torch.tensor(trajectories['states_2dof'][idx], device=self.device)
        a3 = torch.tensor(trajectories['actions_3dof'][idx], device=self.device)
        a2 = torch.tensor(trajectories['actions_2dof'][idx], device=self.device)

        s3_pred = self.state_mapper(s2)
        a2_pred = self.action_mapper(s2, a3)

        action_mse = self.criterion(a2_pred, a2).item()

        # Erreur d'effecteur terminal : on compare l'EE du 3dof prédit avec l'EE réel du 2dof
        ee_error = 0.0
        n_eval = min(512, len(s2))
        for i in range(n_eval):
            ee_pred = ee_from_state(s3_pred[i].cpu().numpy(), 3)   # depuis l'état 3dof prédit
            ee_real = ee_from_state(s2[i].cpu().numpy(), 2)        # depuis l'état 2dof réel
            ee_error += np.linalg.norm(ee_pred - ee_real) ** 2
        ee_error /= n_eval

        return {
            'action_mse': action_mse,
            'ee_error': float(ee_error)
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

    traj_path = data_dir / "trajectories_aligned.pkl"
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)

    print(f"Loaded {trajectories['metadata']['n_samples']} samples")

    # Création des dossiers de logs
    runs_dir = Path("./data/DIRECT/mappers_logs")
    runs_dir.mkdir(exist_ok=True)

    t23 = Transfer2to3(device=device, log_dir=str(runs_dir / "transfer_2to3"))
    t23.train(trajectories, epochs=1200, patience=50)
    t23.save(data_dir / "transfer_2to3.pt")

    t32 = Transfer3to2(device=device, log_dir=str(runs_dir / "transfer_3to2"))
    t32.train(trajectories, epochs=1200, patience=60)
    t32.save(data_dir / "transfer_3to2.pt")

    print("\n=== ENTRAÎNEMENT TERMINÉ ===")
    print("→ Lance tensorboard avec : tensorboard --logdir ./data")


if __name__ == "__main__":
    main()
