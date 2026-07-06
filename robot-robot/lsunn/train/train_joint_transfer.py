"""
Pipeline 4 phases strictement séquentielles.
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lsunn.train.trajectories import collect_reaching_trajectories
from lsunn.tests.transfer_test import test_transfer_2to3, test_transfer_3to2
from lsunn.bases_vae import BaseVAE, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM
from lsunn.unn_policy import UNNPolicy

DATA_DIR = Path("./data/LSUNN")


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("LS-UNN: Joint Transfer Pipeline (4 phases)")
    print("="*60)
    
    # ── Phase 1: Trajectoires IK ──────────────────────────────────────────
    print("\n[Phase 1] Generating IK trajectories...")
    data = collect_reaching_trajectories()
    with open(DATA_DIR / "trajectories.pkl", "wb") as f:
        pickle.dump(data, f)
    
    # ── Phase 2: VAE Bases ────────────────────────────────────────────────
    print("\n[Phase 2] Training VAE bases...")
    import train_vae as tv
    tv.main()
    
    # ── Phase 3: UNN Policies ─────────────────────────────────────────────
    print("\n[Phase 3] Training UNN policies...")
    import train_ppo as tp
    tp.main()
    
    # ── Phase 4: Transfer Tests ───────────────────────────────────────────
    print("\n[Phase 4] Running transfer tests...")
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    
    # Charger les VAE
    base_2dof = BaseVAE(6, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM).to(DEVICE)
    base_3dof = BaseVAE(8, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM).to(DEVICE)
    base_2dof.load_state_dict(torch.load(DATA_DIR / "base_2dof.pt", map_location=DEVICE))
    base_3dof.load_state_dict(torch.load(DATA_DIR / "base_3dof.pt", map_location=DEVICE))
    base_2dof.eval()
    base_3dof.eval()
    
    # Transfer 2→3
    print("\n  Loading 2DoF UNN policy...")
    ppo_2dof = PPO.load(DATA_DIR / "unn_2dof" / "policy.zip", device=DEVICE)
    vec_norm_2dof = VecNormalize.load(DATA_DIR / "unn_2dof" / "vec_normalize.pkl")
    
    # Créer UNNPolicy source avec son propre VAE (pour l'encodage/décodage pendant l'entraînement)
    # Mais pour le transfert, on utilise target_base pour Bi et Bo
    unn_2dof = UNNPolicy(base_2dof, ppo_2dof, vec_norm_2dof, device=DEVICE)
    rate_2to3 = test_transfer_2to3(unn_2dof, base_3dof, DEVICE, n_episodes=100)
    
    # Transfer 3→2
    print("\n  Loading 3DoF UNN policy...")
    ppo_3dof = PPO.load(DATA_DIR / "unn_3dof" / "policy.zip", device=DEVICE)
    vec_norm_3dof = VecNormalize.load(DATA_DIR / "unn_3dof" / "vec_normalize.pkl")
    
    unn_3dof = UNNPolicy(base_3dof, ppo_3dof, vec_norm_3dof, device=DEVICE)
    rate_3to2 = test_transfer_3to2(unn_3dof, base_2dof, DEVICE, n_episodes=100)
    
    print("\n" + "="*60)
    print("Final Transfer Results")
    print("="*60)
    print(f"  Transfer 2DoF → 3DoF: {rate_2to3:.1f}%")
    print(f"  Transfer 3DoF → 2DoF: {rate_3to2:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()