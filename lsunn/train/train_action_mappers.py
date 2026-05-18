"""
Train LatentActionMapper for cross-robot transfer:
  - 2DoF → 3DoF
  - 3DoF → 2DoF

Uses paired trajectories collected during the base VAE phase.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lsunn.unn_policy import LatentActionMapper

# Dimensions
ARM_DIM_2DOF = 6
ARM_DIM_3DOF = 8
ACTION_DIM_2DOF = 2
ACTION_DIM_3DOF = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
EPOCHS = 200
LR = 1e-3

DATA_PATH = Path("./data/LSUNN/trajectories_pushball.pkl")
SAVE_DIR = Path("./data/LSUNN/action_mappers")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_trajectories():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # Les trajectoires sont des listes d'array (n_pairs, n_steps, dim)
    # On concatène toutes les paires et tous les pas de temps
    states_2dof = np.concatenate(data['states_2dof'], axis=0)   # (N, 10)
    states_3dof = np.concatenate(data['states_3dof'], axis=0)   # (N, 12)
    actions_2dof = np.concatenate(data['actions_2dof'], axis=0) # (N, 2)
    actions_3dof = np.concatenate(data['actions_3dof'], axis=0) # (N, 3)

    # Extraire seulement les parties "arm" des états
    arm_2dof = states_2dof[:, :ARM_DIM_2DOF]   # (N, 6)
    arm_3dof = states_3dof[:, :ARM_DIM_3DOF]   # (N, 8)

    return {
        'arm_2dof': arm_2dof,
        'arm_3dof': arm_3dof,
        'act_2dof': actions_2dof,
        'act_3dof': actions_3dof,
    }


def train_mapper_2to3(data, save_path):
    """Mapper: (state_3dof_arm, action_2dof) -> action_3dof"""
    print("\n--- Training 2DoF → 3DoF action mapper ---")
    mapper = LatentActionMapper(
        state_dim=ARM_DIM_3DOF,
        src_action_dim=ACTION_DIM_2DOF,
        tgt_action_dim=ACTION_DIM_3DOF,
    ).to(DEVICE)

    # Préparation des données
    X_state = torch.tensor(data['arm_3dof'], dtype=torch.float32, device=DEVICE)
    X_action = torch.tensor(data['act_2dof'], dtype=torch.float32, device=DEVICE)
    y = torch.tensor(data['act_3dof'], dtype=torch.float32, device=DEVICE)

    dataset = torch.utils.data.TensorDataset(X_state, X_action, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(mapper.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for s, a_src, a_tgt in loader:
            pred = mapper(s, a_src)
            loss = criterion(pred, a_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.6f}")

    torch.save(mapper.state_dict(), save_path)
    print(f"  Saved to {save_path}")
    return mapper


def train_mapper_3to2(data, save_path):
    """Mapper: (state_2dof_arm, action_3dof) -> action_2dof"""
    print("\n--- Training 3DoF → 2DoF action mapper ---")
    mapper = LatentActionMapper(
        state_dim=ARM_DIM_2DOF,
        src_action_dim=ACTION_DIM_3DOF,
        tgt_action_dim=ACTION_DIM_2DOF,
    ).to(DEVICE)

    X_state = torch.tensor(data['arm_2dof'], dtype=torch.float32, device=DEVICE)
    X_action = torch.tensor(data['act_3dof'], dtype=torch.float32, device=DEVICE)
    y = torch.tensor(data['act_2dof'], dtype=torch.float32, device=DEVICE)

    dataset = torch.utils.data.TensorDataset(X_state, X_action, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(mapper.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for s, a_src, a_tgt in loader:
            pred = mapper(s, a_src)
            loss = criterion(pred, a_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.6f}")

    torch.save(mapper.state_dict(), save_path)
    print(f"  Saved to {save_path}")
    return mapper


def main():
    print("Loading paired trajectories...")
    data = load_trajectories()
    print(f"  Total samples: {len(data['arm_2dof']):,}")

    # 2DoF → 3DoF
    train_mapper_2to3(data, SAVE_DIR / "mapper_2to3.pt")

    # 3DoF → 2DoF
    train_mapper_3to2(data, SAVE_DIR / "mapper_3to2.pt")

    print("\n✅ All action mappers trained and saved.")


if __name__ == "__main__":
    main()
