# lsunn/train/train_vae.py
"""
Phase 2 : Entraînement des BaseVAE (2DoF et 3DoF) dans un espace latent partagé.
Charge les trajectoires depuis DATA_DIR/trajectories.pkl.
Sauvegarde les poids dans DATA_DIR/{base_2dof,base_3dof}.pt.
"""

import sys
import numpy as np
import pickle
import gc
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lsunn.bases_vae import BaseVAE, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("./data/LSUNN/joint_model")
TRAJ_PATH = DATA_DIR / "trajectories.pkl"

LATENT_DIM = 16
HIDDEN_DIM = 128   # réduit pour économiser la mémoire


# ── Entraînement ──────────────────────────────────────────────────────────────

def train_vae(
    data: dict,
    base_2dof: BaseVAE,
    base_3dof: BaseVAE,
    device: str = "cpu",
    epochs: int = 150,
) -> None:
    """
    Entraîne les deux VAE en partageant l'espace latent grâce à :
      - une perte de reconstruction (MSE)
      - une régularisation KL
      - une perte d'alignement latent entre z2 et z3
      - une perte de reconstruction croisée (décodeur de l'autre VAE)
    """
    print("\n" + "=" * 60)
    print("Phase 2: Training VAE bases (shared latent space)")
    print("=" * 60)

    n_samples = min(50_000, len(data["states_2dof"]))
    indices   = np.random.choice(len(data["states_2dof"]), n_samples, replace=False)

    s2 = torch.tensor(data["states_2dof"][indices], dtype=torch.float32, device=device)
    s3 = torch.tensor(data["states_3dof"][indices], dtype=torch.float32, device=device)

    optimizer   = optim.Adam(
        list(base_2dof.parameters()) + list(base_3dof.parameters()), lr=3e-4
    )
    recon_crit = nn.MSELoss()
    sim_crit   = nn.MSELoss()

    pbar = tqdm(range(epochs), desc="VAE Training")
    for _ in pbar:
        perm       = torch.randperm(len(s2))
        epoch_loss = 0.0

        for i in range(0, len(s2), 256):
            idx = perm[i : i + 256]
            b2, b3 = s2[idx], s3[idx]

            z2, mu2, logvar2 = base_2dof.encoder(b2)
            z3, mu3, logvar3 = base_3dof.encoder(b3)

            recon2 = base_2dof.decoder(z2)
            recon3 = base_3dof.decoder(z3)
            cross2 = base_2dof.decoder(z3)   # décode z3 avec le décodeur 2DoF
            cross3 = base_3dof.decoder(z2)   # décode z2 avec le décodeur 3DoF

            kl2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp()) / b2.shape[0]
            kl3 = -0.5 * torch.sum(1 + logvar3 - mu3.pow(2) - logvar3.exp()) / b3.shape[0]

            loss = (
                10.0 * (recon_crit(recon2, b2) + recon_crit(recon3, b3))
                + 0.001 * (kl2 + kl3)
                + 1.0 * sim_crit(z2, z3)
                + 1.0 * (recon_crit(cross2, b2) + recon_crit(cross3, b3))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        denom = max(1, (len(s2) + 255) // 256)
        pbar.set_postfix({"loss": f"{epoch_loss / denom:.4f}"})

    print("  VAE training complete")

    del s2, s3
    gc.collect()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DEVICE = "cpu"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Charger les trajectoires ───────────────────────────────────────────
    print(f"  Loading trajectories from {TRAJ_PATH} …")
    with open(TRAJ_PATH, "rb") as f:
        data = pickle.load(f)

    obs_dim_2dof = data["states_2dof"].shape[1]
    obs_dim_3dof = data["states_3dof"].shape[1]

    # ── Instancier les VAE ─────────────────────────────────────────────────
    base_2dof = BaseVAE(obs_dim_2dof, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
    base_3dof = BaseVAE(obs_dim_3dof, LATENT_DIM, HIDDEN_DIM).to(DEVICE)

    # ── Entraînement ──────────────────────────────────────────────────────
    train_vae(data, base_2dof, base_3dof, device=DEVICE, epochs=150)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    torch.save(base_2dof.state_dict(), DATA_DIR / "base_2dof.pt")
    torch.save(base_3dof.state_dict(), DATA_DIR / "base_3dof.pt")
    print(f"  VAE weights saved → {DATA_DIR}/base_{{2,3}}dof.pt")


if __name__ == "__main__":
    main()
