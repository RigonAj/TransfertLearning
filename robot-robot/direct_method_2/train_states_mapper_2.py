import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# --- Configuration avec les DIMENSIONS CORRIGÉES ---
R1_NAME = "robot_2dofs"
R2_NAME = "robot_3dofs"

# Dimensions d'état CORRECTES (correspondent à Arm2DoF et Arm3DoF)
STATE_DIM_R1 = 6  # [θ1, θ2, dθ1, dθ2, x, y]
STATE_DIM_R2 = 8  # [θ1, θ2, θ3, dθ1, dθ2, dθ3, x, y]
HIDDEN_DIM = 256

PATH_R1_TRAJ = f"direct_method_2/trajectories/{R1_NAME}.txt"
PATH_R2_TRAJ = f"direct_method_2/trajectories/{R2_NAME}.txt"

NUM_EPOCHS = 100
LEARNING_RATE = 0.003
BATCH_SIZE = 100
RUN_ID = "run_01"

# --- Output dirs ---
RUN_DIR = os.path.join("direct_method_2/runs", RUN_ID)
MODEL_DIR = os.path.join(RUN_DIR, "models")
LOG_DIR = os.path.join(RUN_DIR, "logs")
PLOT_DIR = os.path.join(RUN_DIR, "plots")
for d in (MODEL_DIR, LOG_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)

# ==============================================
# 1. DATASET CORRIGÉ (intégré ici pour simplifier)
# ==============================================
class TrajectoriesTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories_r1,
        trajectories_r2,
        dim_R1=6,
        dim_R2=8,
        nq_r1=2,
        nq_r2=3,
        omega_max=2.0,
        max_reach=3.0,
    ):
        # Chargement
        self.trajectories_r1 = np.loadtxt(trajectories_r1, dtype=np.float32)[:, :dim_R1]
        self.trajectories_r2 = np.loadtxt(trajectories_r2, dtype=np.float32)[:, :dim_R2]

        # Extraction des vitesses
        vel_r1 = self.trajectories_r1[:, nq_r1:2*nq_r1]
        vel_r2 = self.trajectories_r2[:, nq_r2:2*nq_r2]

        # Masque de validité
        valid_mask_r1 = np.all((vel_r1 >= -omega_max) & (vel_r1 <= omega_max), axis=1)
        valid_mask_r2 = np.all((vel_r2 >= -omega_max) & (vel_r2 <= omega_max), axis=1)
        valid_mask = valid_mask_r1 & valid_mask_r2

        self.trajectories_r1 = self.trajectories_r1[valid_mask]
        self.trajectories_r2 = self.trajectories_r2[valid_mask]

        # Normalisation
        # Positions articulaires / Pi
        self.trajectories_r1[:, :nq_r1] /= np.pi
        self.trajectories_r2[:, :nq_r2] /= np.pi
        # Vitesses articulaires / Omega Max
        self.trajectories_r1[:, nq_r1:2*nq_r1] /= omega_max
        self.trajectories_r2[:, nq_r2:2*nq_r2] /= omega_max
        # Position de l'effecteur / Max Reach
        self.trajectories_r1[:, 2*nq_r1:] /= max_reach
        self.trajectories_r2[:, 2*nq_r2:] /= max_reach

    def __len__(self):
        return len(self.trajectories_r1)

    def __getitem__(self, idx):
        return (self.trajectories_r1[idx], self.trajectories_r2[idx])

# ==============================================
# 2. MODÈLE MAPPER CORRIGÉ
# ==============================================
class StateMapperMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ==============================================
# 3. SCRIPT PRINCIPAL
# ==============================================
if __name__ == "__main__":
    # Dataset
    full_dataset = TrajectoriesTrainingDataset(
        PATH_R1_TRAJ, PATH_R2_TRAJ,
        dim_R1=STATE_DIM_R1, dim_R2=STATE_DIM_R2,
        nq_r1=2, nq_r2=3,
    )
    total_size = len(full_dataset)
    train_size = int(0.85 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size
    data_train, data_val, data_test = random_split(full_dataset, [train_size, val_size, test_size])
    del full_dataset

    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # Modèles avec les DIMENSIONS CORRIGÉES
    model_r1_to_r2 = StateMapperMLP(STATE_DIM_R1, STATE_DIM_R2, HIDDEN_DIM)
    model_r2_to_r1 = StateMapperMLP(STATE_DIM_R2, STATE_DIM_R1, HIDDEN_DIM)

    mse_loss = nn.MSELoss()
    optimizer_r1_to_r2 = torch.optim.Adam(model_r1_to_r2.parameters(), lr=LEARNING_RATE)
    optimizer_r2_to_r1 = torch.optim.Adam(model_r2_to_r1.parameters(), lr=LEARNING_RATE)

    history = {"train_r1_to_r2": [], "val_r1_to_r2": [], "train_r2_to_r1": [], "val_r2_to_r1": []}

    # CSV log
    LOG_PATH = os.path.join(LOG_DIR, "training_log.csv")
    with open(LOG_PATH, "w", newline="") as log_file:
        log_writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_r1_to_r2", "val_r1_to_r2", "train_r2_to_r1", "val_r2_to_r1"])
        log_writer.writeheader()

        pbar = tqdm(range(NUM_EPOCHS), desc="Training State Mappers")

        for epoch in pbar:
            # Entraînement
            model_r1_to_r2.train()
            model_r2_to_r1.train()
            train_sum_r1_to_r2 = 0.0
            train_sum_r2_to_r1 = 0.0
            for state_r1, state_r2 in train_loader:
                # r1 → r2
                batch_loss = mse_loss(model_r1_to_r2(state_r1), state_r2)
                optimizer_r1_to_r2.zero_grad()
                batch_loss.backward()
                optimizer_r1_to_r2.step()
                train_sum_r1_to_r2 += batch_loss.item()
                # r2 → r1
                batch_loss = mse_loss(model_r2_to_r1(state_r2), state_r1)
                optimizer_r2_to_r1.zero_grad()
                batch_loss.backward()
                optimizer_r2_to_r1.step()
                train_sum_r2_to_r1 += batch_loss.item()

            n = len(train_loader)
            train_r1_to_r2 = train_sum_r1_to_r2 / n
            train_r2_to_r1 = train_sum_r2_to_r1 / n
            history["train_r1_to_r2"].append(train_r1_to_r2)
            history["train_r2_to_r1"].append(train_r2_to_r1)

            # Validation
            model_r1_to_r2.eval()
            model_r2_to_r1.eval()
            val_sum_r1_to_r2 = 0.0
            val_sum_r2_to_r1 = 0.0
            with torch.no_grad():
                for state_r1, state_r2 in val_loader:
                    val_sum_r1_to_r2 += mse_loss(model_r1_to_r2(state_r1), state_r2).item()
                    val_sum_r2_to_r1 += mse_loss(model_r2_to_r1(state_r2), state_r1).item()
            val_r1_to_r2 = val_sum_r1_to_r2 / len(val_loader)
            val_r2_to_r1 = val_sum_r2_to_r1 / len(val_loader)
            history["val_r1_to_r2"].append(val_r1_to_r2)
            history["val_r2_to_r1"].append(val_r2_to_r1)

            # Logs
#            print(f"Epoch [{epoch+1:>3}/{NUM_EPOCHS}]  train({train_r1_to_r2:.6f} / {train_r2_to_r1:.6f})  val({val_r1_to_r2:.6f} / {val_r2_to_r1:.6f})")
            log_writer.writerow({"epoch": epoch+1, "train_r1_to_r2": train_r1_to_r2, "val_r1_to_r2": val_r1_to_r2, "train_r2_to_r1": train_r2_to_r1, "val_r2_to_r1": val_r2_to_r1})
            pbar.set_postfix({"Epoch": f"[{epoch+1:>3}/{NUM_EPOCHS}]", 
                              "train": f"({train_r1_to_r2:.6f}/{train_r2_to_r1:.6f})", 
                              "val": f"({val_r1_to_r2:.6f}/{val_r2_to_r1:.6f})"
                              })

    # Sauvegarde des modèles
    for name, model in [("state_mapper_r1_to_r2", model_r1_to_r2), ("state_mapper_r2_to_r1", model_r2_to_r1)]:
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        torch.save(model.state_dict(), path)
        print(f"Model saved → {path}")

    print("\nDone. All outputs in:", RUN_DIR)