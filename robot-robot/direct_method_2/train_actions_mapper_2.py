import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from direct_method_2.dataset_2 import TrajectoriesTrainingActionDataset
from direct_method_2.mapper_models_2 import ActionMapperMLP

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
R1_NAME        = "robot_2dofs"
R2_NAME        = "robot_3dofs"
NQ_R1          = 2
NQ_R2          = 3
ACTION_DIM_R1  = 2
ACTION_DIM_R2  = 3
HIDDEN_DIM     = 256

PATH_R1_TRAJ = f"direct_method/trajectories/{R1_NAME}.txt"
PATH_R2_TRAJ = f"direct_method/trajectories/{R2_NAME}.txt"

NUM_EPOCHS    = 100
LEARNING_RATE = 0.003
BATCH_SIZE    = 100
RUN_ID        = "run_02_cond"

# Output dirs
RUN_DIR   = os.path.join("direct_method/runs", RUN_ID)
MODEL_DIR = os.path.join(RUN_DIR, "models")
LOG_DIR   = os.path.join(RUN_DIR, "logs")
PLOT_DIR  = os.path.join(RUN_DIR, "plots")
for d in (MODEL_DIR, LOG_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset & splits
# ──────────────────────────────────────────────
full_dataset = TrajectoriesTrainingActionDataset(
    PATH_R1_TRAJ, PATH_R2_TRAJ,
    nq_r1=NQ_R1, nq_r2=NQ_R2
)

total_size = len(full_dataset)
train_size = int(0.85 * total_size)
val_size   = int(0.10 * total_size)
test_size  = total_size - train_size - val_size

data_train, data_val, data_test = random_split(
    full_dataset, [train_size, val_size, test_size]
)
del full_dataset

train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8)
val_loader   = DataLoader(data_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
test_loader  = DataLoader(data_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
model_r1_to_r2 = ActionMapperMLP(ACTION_DIM_R1, ACTION_DIM_R2, q_dim=NQ_R1, hidden=HIDDEN_DIM)
model_r2_to_r1 = ActionMapperMLP(ACTION_DIM_R2, ACTION_DIM_R1, q_dim=NQ_R2, hidden=HIDDEN_DIM)

mse_loss = nn.MSELoss()

optimizer_r1_to_r2 = torch.optim.Adam(model_r1_to_r2.parameters(), lr=LEARNING_RATE)
optimizer_r2_to_r1 = torch.optim.Adam(model_r2_to_r1.parameters(), lr=LEARNING_RATE)

# ──────────────────────────────────────────────
# History & logging
# ──────────────────────────────────────────────
history = {
    "train_r1_to_r2": [], "val_r1_to_r2": [],
    "train_r2_to_r1": [], "val_r2_to_r1": [],
}

LOG_PATH   = os.path.join(LOG_DIR, "training_log.csv")
CSV_FIELDS = ["epoch", "train_r1_to_r2", "val_r1_to_r2", "train_r2_to_r1", "val_r2_to_r1"]

log_file   = open(LOG_PATH, "w", newline="")
log_writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
log_writer.writeheader()

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def evaluate_val(loader):
    """Compute mean MSE over full validation set."""
    sum_r1_to_r2 = 0.0
    sum_r2_to_r1 = 0.0
    n = 0
    with torch.no_grad():
        for vel_r1, q_r1, vel_r2, q_r2 in loader:
            sum_r1_to_r2 += mse_loss(model_r1_to_r2(vel_r1, q_r1), vel_r2).item()
            sum_r2_to_r1 += mse_loss(model_r2_to_r1(vel_r2, q_r2), vel_r1).item()
            n += 1
    return sum_r1_to_r2 / n, sum_r2_to_r1 / n


def save_plot(history, path):
    """Save train/val loss curves for both directions."""
    epochs = range(1, len(history["train_r1_to_r2"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

    ax1.plot(epochs, history["train_r1_to_r2"], label="train", color="steelblue",  linewidth=1.8)
    ax1.plot(epochs, history["val_r1_to_r2"],   label="val",   color="darkorange", linewidth=1.8, linestyle="--")
    ax1.set_title("MSE  2DoF → 3DoF")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.legend()
    ax1.grid(True, alpha=0.35)

    ax2.plot(epochs, history["train_r2_to_r1"], label="train", color="steelblue",  linewidth=1.8)
    ax2.plot(epochs, history["val_r2_to_r1"],   label="val",   color="darkorange", linewidth=1.8, linestyle="--")
    ax2.set_title("MSE  3DoF → 2DoF")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.legend()
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")

# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
if __name__ == "__main__":
    pbar = tqdm(range(NUM_EPOCHS), desc="Training Conditioned Actions Mappers")

    for epoch in pbar:
        # ── Train ─────────────────────────────
        model_r1_to_r2.train()
        model_r2_to_r1.train()

        train_sum_r1_to_r2 = 0.0
        train_sum_r2_to_r1 = 0.0

        for vel_r1, q_r1, vel_r2, q_r2 in train_loader:
            # 2DoF → 3DoF
            batch_loss = mse_loss(model_r1_to_r2(vel_r1, q_r1), vel_r2)
            optimizer_r1_to_r2.zero_grad()
            batch_loss.backward()
            optimizer_r1_to_r2.step()
            train_sum_r1_to_r2 += batch_loss.item()

            # 3DoF → 2DoF
            batch_loss = mse_loss(model_r2_to_r1(vel_r2, q_r2), vel_r1)
            optimizer_r2_to_r1.zero_grad()
            batch_loss.backward()
            optimizer_r2_to_r1.step()
            train_sum_r2_to_r1 += batch_loss.item()

        n = len(train_loader)
        train_r1_to_r2 = train_sum_r1_to_r2 / n
        train_r2_to_r1 = train_sum_r2_to_r1 / n

        history["train_r1_to_r2"].append(train_r1_to_r2)
        history["train_r2_to_r1"].append(train_r2_to_r1)

        # ── Validation ────────────────────────
        model_r1_to_r2.eval()
        model_r2_to_r1.eval()

        val_r1_to_r2, val_r2_to_r1 = evaluate_val(val_loader)

        history["val_r1_to_r2"].append(val_r1_to_r2)
        history["val_r2_to_r1"].append(val_r2_to_r1)

        # ── Logging ───────────────────────────
        log_writer.writerow({
            "epoch":           epoch + 1,
            "train_r1_to_r2":  train_r1_to_r2,
            "val_r1_to_r2":    val_r1_to_r2,
            "train_r2_to_r1":  train_r2_to_r1,
            "val_r2_to_r1":    val_r2_to_r1,
        })
        log_file.flush()

        pbar.set_postfix({
            "Epoch": f"[{epoch+1:>3}/{NUM_EPOCHS}]",
            "train": f"({train_r1_to_r2:.6f}/{train_r2_to_r1:.6f})",
            "val":   f"({val_r1_to_r2:.6f}/{val_r2_to_r1:.6f})",
        })

    # ── Qualitative sample ────────────────────
    with torch.no_grad():
        idx = np.random.randint(0, len(data_val))
        v_r1, q_r1, v_r2, q_r2 = data_val[idx]

        pred_r2 = model_r1_to_r2(
            torch.from_numpy(v_r1).unsqueeze(0), 
            torch.from_numpy(q_r1).unsqueeze(0)
        ).squeeze(0)

        pred_r1 = model_r2_to_r1(
            torch.from_numpy(v_r2).unsqueeze(0), 
            torch.from_numpy(q_r2).unsqueeze(0)
        ).squeeze(0)

        print("─" * 100)
        print(f"  [2DoF→3DoF] target: {[round(v, 4) for v in v_r2.tolist()]}")
        print(f"              pred:   {[round(v, 4) for v in pred_r2.tolist()]}")
        print(f"              q_2dof: {[round(v, 4) for v in q_r1.tolist()]}")
        print(f"  [3DoF→2DoF] target: {[round(v, 4) for v in v_r1.tolist()]}")
        print(f"              pred:   {[round(v, 4) for v in pred_r1.tolist()]}")
        print(f"              q_3dof: {[round(v, 4) for v in q_r2.tolist()]}")
        print(f"  Pred std 2→3: {pred_r2.std().item():.4f}  Target std: {v_r2.std().item():.4f}")
        print("─" * 100)

    # ── Save models ───────────────────────────
    log_file.close()

    for name, model in [
        ("action_mapper_2dof_to_3dof", model_r1_to_r2),
        ("action_mapper_3dof_to_2dof", model_r2_to_r1),
    ]:
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        torch.save(model.state_dict(), path)
        print(f"Model saved → {path}")

    # ── Save plot ─────────────────────────────
    plot_path = os.path.join(PLOT_DIR, "loss_curves.png")
    save_plot(history, plot_path)

    print(f"\nDone. All outputs in: {RUN_DIR}")
    print("  models/ — 2dof_to_3dof.pt, 3dof_to_2dof.pt")
    print("  logs/   — training_log.csv")
    print("  plots/  — loss_curves.png")