"""
LS-UNN Bases VAE — Variational Autoencoders for shared latent space alignment.
Based on: "Towards Zero-Shot Cross-Agent Transfer Learning via
Latent-Space Universal Notice Network" (Beaussant et al., 2024)

Architecture:
  - Each robot has a VAE (encoder + decoder) that maps its full state
    (arm_obs + task_obs) to/from a shared latent space.
  - The two VAEs are trained jointly with:
      * Reconstruction loss (MSE)
      * KL divergence loss (β-VAE regularization)
      * Latent similarity loss (MSE between z₁ and z₂)
      * Cross-reconstruction loss (decode z₂ with base_r1 and vice versa)
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm

torch.set_num_threads(8)

# ============================================================================
# Dimensions
# ============================================================================
STATE_DIM_2DOF = 10
STATE_DIM_3DOF = 12
ACTION_DIM_2DOF = 2
ACTION_DIM_3DOF = 3

DEFAULT_LATENT_DIM = 16
DEFAULT_HIDDEN_DIM = 256


# ============================================================================
# VAE Encoder Base (Input Base)
# ============================================================================
class VAEEncoder(nn.Module):
    """Maps state → (μ, log_σ²) in latent space."""
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


# ============================================================================
# VAE Decoder Base (Output Base)
# ============================================================================
class VAEDecoder(nn.Module):
    """Maps latent z → reconstructed state."""
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ============================================================================
# Full VAE (Encoder + Decoder) for one robot
# ============================================================================
class BaseVAE(nn.Module):
    """Full VAE for a single robot: encodes state → latent z, decodes z → state."""
    def __init__(self, state_dim: int, latent_dim: int = DEFAULT_LATENT_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(state_dim, latent_dim, hidden_dim)
        self.decoder = VAEDecoder(state_dim, latent_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic encoding (use μ only, no sampling)."""
        z, _, _ = self.encoder(x)
        return z

    @torch.no_grad()
    def encode_np(self, x: np.ndarray, device: str = "cuda") -> np.ndarray:
        """Encode numpy array to latent numpy array."""
        t = torch.tensor(x, dtype=torch.float32, device=device)
        return self.encode(t).cpu().numpy()

    @torch.no_grad()
    def decode_np(self, z: np.ndarray, device: str = "cuda") -> np.ndarray:
        """Decode latent numpy array to state numpy array."""
        t = torch.tensor(z, dtype=torch.float32, device=device)
        return self.decoder(t).cpu().numpy()


# ============================================================================
# VAE Trainer — Joint training of both bases
# ============================================================================
class BasesVAETrainer:
    """
    Trains two VAEs (one per robot) to learn a shared latent space.

    Loss = α·(L_recon_r1 + L_recon_r2)           # reconstruction
         + β·(L_KL_r1 + L_KL_r2)                 # KL regularization
         + γ·L_sim(z₁, z₂)                       # latent similarity
         + λ·(L_cross_r1 + L_cross_r2)            # cross-reconstruction
    """
    def __init__(self, state_dim_r1: int, state_dim_r2: int,
                 latent_dim: int = DEFAULT_LATENT_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 device: str = "cpu"):
        self.device = device
        self.latent_dim = latent_dim

        self.base_r1 = BaseVAE(state_dim_r1, latent_dim, hidden_dim).to(device)
        self.base_r2 = BaseVAE(state_dim_r2, latent_dim, hidden_dim).to(device)

        self.reconstruction_crit = nn.MSELoss()
        self.similarity_crit = nn.MSELoss()

    def _split(self, *tensors, val_frac=0.1):
        n = len(tensors[0])
        idx = torch.randperm(n)
        cut = int(n * (1 - val_frac))
        tr, vl = idx[:cut], idx[cut:]
        return tuple(t[tr] for t in tensors) + tuple(t[vl] for t in tensors)

    def train(self, trajectories: Dict, epochs: int = 300,
              batch_size: int = 512, lr: float = 3e-4,
              alpha: float = 10.0, beta: float = 0.001,
              gamma: float = 1.0, lmbda: float = 1.0):
        """
        Train both VAEs jointly.

        Args:
            trajectories: dict with 'states_2dof', 'states_3dof' (lists of arrays)
            alpha: reconstruction weight
            beta: KL divergence weight
            gamma: latent similarity weight
            lmbda: cross-reconstruction weight
        """
        print("\n" + "="*60)
        print("LS-UNN: Joint VAE Bases Training")
        print("="*60)
        print(f"  Device: {self.device}")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  α(recon)={alpha}  β(KL)={beta}  γ(sim)={gamma}  λ(cross)={lmbda}")

        s_r1 = torch.tensor(
            np.concatenate(trajectories['states_2dof'], axis=0), dtype=torch.float32
        ).to(self.device)
        s_r2 = torch.tensor(
            np.concatenate(trajectories['states_3dof'], axis=0), dtype=torch.float32
        ).to(self.device)

        print(f"  Dataset: {len(s_r1):,} paired samples\n")

        splits = self._split(s_r1, s_r2)
        s1_tr, s2_tr, s1_vl, s2_vl = splits

        optimizer = optim.Adam(
            list(self.base_r1.parameters()) + list(self.base_r2.parameters()),
            lr=lr, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        best_val = float('inf')
        best_state = None

        pbar = tqdm(range(epochs), desc="Training VAE Bases")

        for epoch in pbar:
            self.base_r1.train()
            self.base_r2.train()

            perm = torch.randperm(len(s1_tr))
            total_loss, n_b = 0.0, 0

            for i in range(0, len(s1_tr), batch_size):
                idx = perm[i:i + batch_size]
                b1, b2 = s1_tr[idx], s2_tr[idx]

                # --- Forward ---
                z1, mu1, logvar1 = self.base_r1.encoder(b1)
                z2, mu2, logvar2 = self.base_r2.encoder(b2)

                y1 = self.base_r1.decoder(z1)
                y2 = self.base_r2.decoder(z2)

                y_cross_1 = self.base_r1.decoder(z2)
                y_cross_2 = self.base_r2.decoder(z1)

                # --- Losses ---
                loss_recon = alpha * (
                    self.reconstruction_crit(y1, b1) +
                    self.reconstruction_crit(y2, b2)
                )

                loss_kl = beta * (
                    -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp()) / b1.shape[0] +
                    -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp()) / b2.shape[0]
                )

                loss_sim = gamma * self.similarity_crit(z1, z2)

                loss_cross = lmbda * (
                    self.reconstruction_crit(y_cross_1, b1) +
                    self.reconstruction_crit(y_cross_2, b2)
                )

                loss = loss_recon + loss_kl + loss_sim + loss_cross

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_b += 1

            scheduler.step()

            # Validation
            if (epoch + 1) % 25 == 0:
                self.base_r1.eval()
                self.base_r2.eval()
                with torch.no_grad():
                    z1_v, _, _ = self.base_r1.encoder(s1_vl)
                    z2_v, _, _ = self.base_r2.encoder(s2_vl)
                    val_loss = self.similarity_crit(z1_v, z2_v).item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {
                        'r1': {k: v.clone() for k, v in self.base_r1.state_dict().items()},
                        'r2': {k: v.clone() for k, v in self.base_r2.state_dict().items()},
                    }

                pbar.set_postfix({
                    'train_loss': f'{total_loss/n_b:.4f}',
                    'val_sim': f'{val_loss:.6f}',
                    'best_val': f'{best_val:.6f}',
                })

        if best_state is not None:
            self.base_r1.load_state_dict(best_state['r1'])
            self.base_r2.load_state_dict(best_state['r2'])

        print(f"\n  Training complete (best val similarity: {best_val:.6f})")

    def save(self, save_dir: str, run_id: str = "pushball_bases"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.base_r1.state_dict(), save_path / f"{run_id}_r1_2dof.pt")
        torch.save(self.base_r2.state_dict(), save_path / f"{run_id}_r2_3dof.pt")

        config = {
            'latent_dim': self.latent_dim,
            'state_dim_r1': self.base_r1.state_dim,
            'state_dim_r2': self.base_r2.state_dim,
        }
        with open(save_path / f"{run_id}_config.pkl", 'wb') as f:
            pickle.dump(config, f)

        print(f"  Bases saved → {save_path}")

    def load(self, save_dir: str, run_id: str = "pushball_bases"):
        save_path = Path(save_dir)
        self.base_r1.load_state_dict(
            torch.load(save_path / f"{run_id}_r1_2dof.pt", map_location=self.device)
        )
        self.base_r2.load_state_dict(
            torch.load(save_path / f"{run_id}_r2_3dof.pt", map_location=self.device)
        )
        print(f"  Bases loaded ← {save_path}")


# ============================================================================
# Main: Train bases from trajectories
# ============================================================================
def main(traj_path: Optional[str] = None, save_dir: str = "./data/LSUNN/bases"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if traj_path is None:
        # Chemin par défaut cohérent avec trajectories_pushball.py
        traj_path = "./data/LSUNN/trajectories_pushball.pkl"
    traj_path = Path(traj_path)

    bases_dir = Path(save_dir)

    print("\n" + "="*60)
    print("LS-UNN: Loading Trajectories")
    print("="*60)
    print(f"  Loading from: {traj_path}")

    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)

    meta = trajectories['metadata']
    print(f"  {meta['n_pairs']} pairs | source: {meta.get('source', '?')}")

    trainer = BasesVAETrainer(
        state_dim_r1=STATE_DIM_2DOF,
        state_dim_r2=STATE_DIM_3DOF,
        latent_dim=DEFAULT_LATENT_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        device=device,
    )

    trainer.train(trajectories, epochs=300, batch_size=512)
    trainer.save(str(bases_dir), run_id="pushball_bases")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(traj_path=sys.argv[1])
    else:
        main()
