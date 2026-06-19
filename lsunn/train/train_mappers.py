# lsunn/train/train_mappers.py
"""
Phase 4 : Entraînement des mappers d'actions bidirectionnels (2DoF↔3DoF).
Charge les trajectoires depuis DATA_DIR/trajectories.pkl.
Sauvegarde les poids dans DATA_DIR/{mapper_2to3,mapper_3to2}.pt.
"""

import sys
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Config ────────────────────────────────────────────────────────────────────
ARM_OBS_2DOF    = 6
ARM_OBS_3DOF    = 8
ACTION_DIM_2DOF = 2
ACTION_DIM_3DOF = 3

DATA_DIR  = Path("./data/LSUNN/joint_model")
TRAJ_PATH = DATA_DIR / "trajectories.pkl"


# ── Modèle ────────────────────────────────────────────────────────────────────

class ActionMapper(nn.Module):
    """
    Mapper d'action conditionné sur l'état du bras.
    Entrée  : [état_bras (state_dim) ‖ action_source (act_in_dim)]
    Sortie  : action_cible (act_out_dim) ∈ [-1, 1]
    """

    def __init__(self, state_dim: int, act_in_dim: int,
                 act_out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim   = state_dim
        self.act_in_dim  = act_in_dim
        self.act_out_dim = act_out_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim + act_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_out_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        assert state.shape[1] == self.state_dim, (
            f"Expected state dim {self.state_dim}, got {state.shape[1]}"
        )
        assert action.shape[1] == self.act_in_dim, (
            f"Expected action dim {self.act_in_dim}, got {action.shape[1]}"
        )

        return self.net(torch.cat([state, action], dim=-1))


# ── Entraînement ──────────────────────────────────────────────────────────────

def _train_single_mapper(
    mapper: ActionMapper,
    states: torch.Tensor,
    actions_in: torch.Tensor,
    actions_out: torch.Tensor,
    epochs: int,
    name: str,
) -> None:
    """Entraîne un mapper avec split train/val (90/10)."""
    optimizer  = optim.Adam(mapper.parameters(), lr=1e-3)
    criterion  = nn.MSELoss()

    n_samples = len(states)
    split     = int(n_samples * 0.9)
    idx       = torch.randperm(n_samples)

    train_s    = states[idx[:split]]
    train_a_in = actions_in[idx[:split]]
    train_a_out = actions_out[idx[:split]]
    val_s      = states[idx[split:]]
    val_a_in   = actions_in[idx[split:]]
    val_a_out  = actions_out[idx[split:]]

    best_val_loss = float("inf")
    pbar = tqdm(range(epochs), desc=name)

    for _ in pbar:
        mapper.train()
        pred = mapper(train_s, train_a_in)
        loss = criterion(pred, train_a_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mapper.eval()
        with torch.no_grad():
            val_pred = mapper(val_s, val_a_in)
            val_loss = criterion(val_pred, val_a_out)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        pbar.set_postfix({
            "train": f"{loss.item():.4f}",
            "val":   f"{val_loss.item():.4f}",
        })

    print(f"    Best val loss: {best_val_loss:.6f}")


def train_mappers(
    data: dict,
    device: str = "cpu",
    epochs: int = 200,
) -> tuple[ActionMapper, ActionMapper]:
    """
    Entraîne les deux mappers bidirectionnels et retourne
    (mapper_2to3, mapper_3to2).
    """
    print("\n" + "=" * 60)
    print("Phase 4: Training action mappers for bidirectional transfer")
    print("=" * 60)

    n_samples = min(50_000, len(data["arm_states_2dof"]))
    indices   = np.random.choice(len(data["arm_states_2dof"]), n_samples, replace=False)

    s2 = torch.tensor(data["arm_states_2dof"][indices], dtype=torch.float32, device=device)
    s3 = torch.tensor(data["arm_states_3dof"][indices], dtype=torch.float32, device=device)
    a2 = torch.tensor(data["actions_2dof"][indices],    dtype=torch.float32, device=device)
    a3 = torch.tensor(data["actions_3dof"][indices],    dtype=torch.float32, device=device)

    # ── Mapper 2→3 ────────────────────────────────────────────────────────
    print(f"\n  Training mapper 2→3  "
          f"[state={ARM_OBS_3DOF}, in={ACTION_DIM_2DOF}, out={ACTION_DIM_3DOF}]")
    mapper_2to3 = ActionMapper(ARM_OBS_3DOF, ACTION_DIM_2DOF, ACTION_DIM_3DOF,
                               hidden_dim=256).to(device)
    _train_single_mapper(mapper_2to3, s3, a2, a3, epochs, "mapper_2to3")

    # ── Mapper 3→2 ────────────────────────────────────────────────────────
    print(f"\n  Training mapper 3→2  "
          f"[state={ARM_OBS_2DOF}, in={ACTION_DIM_3DOF}, out={ACTION_DIM_2DOF}]")
    mapper_3to2 = ActionMapper(ARM_OBS_2DOF, ACTION_DIM_3DOF, ACTION_DIM_2DOF,
                               hidden_dim=256).to(device)
    _train_single_mapper(mapper_3to2, s2, a3, a2, epochs, "mapper_3to2")

    return mapper_2to3, mapper_3to2


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DEVICE = "cpu"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Charger les trajectoires ───────────────────────────────────────────
    print(f"  Loading trajectories from {TRAJ_PATH} …")
    with open(TRAJ_PATH, "rb") as f:
        data = pickle.load(f)

    # ── Entraînement ──────────────────────────────────────────────────────
    mapper_2to3, mapper_3to2 = train_mappers(data, device=DEVICE, epochs=200)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    torch.save(mapper_2to3.state_dict(), DATA_DIR / "mapper_2to3.pt")
    torch.save(mapper_3to2.state_dict(), DATA_DIR / "mapper_3to2.pt")
    print(f"\n  Mapper weights saved → {DATA_DIR}/mapper_{{2to3,3to2}}.pt")


if __name__ == "__main__":
    main()
