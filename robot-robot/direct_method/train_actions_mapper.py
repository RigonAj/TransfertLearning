import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from direct_method.dataset import TrajectoriesTrainingDataset, TrajectoriesTrainingActionDataset
from torch.utils.data import DataLoader, random_split
from direct_method.mapper_models import StateMapperMLP,ActionMapperMLP
from tqdm import tqdm

# Config
R1_NAME      = "robot_2dofs"
R2_NAME      = "robot_3dofs"
ACTION_DIM_R1 = 2
ACTION_DIM_R2 = 3
HIDDEN_DIM   = 256

PATH_R1_TRAJ = "direct_method/trajectories/{}.txt".format(R1_NAME)
PATH_R2_TRAJ = "direct_method/trajectories/{}.txt".format(R2_NAME)

NUM_EPOCHS    = 100
LEARNING_RATE = 0.003
BATCH_SIZE    = 100
RUN_ID        = "run_02"

# Output dirs
RUN_DIR   = os.path.join("direct_method/runs", RUN_ID)
MODEL_DIR = os.path.join(RUN_DIR, "models")
LOG_DIR   = os.path.join(RUN_DIR, "logs")
PLOT_DIR  = os.path.join(RUN_DIR, "plots")

for d in (MODEL_DIR, LOG_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)

# Dataset & splits
full_dataset = TrajectoriesTrainingActionDataset(PATH_R1_TRAJ, PATH_R2_TRAJ, ACTION_DIM_R1, ACTION_DIM_R2)

total_size = len(full_dataset)
train_size = int(0.85 * total_size)
val_size   = int(0.10 * total_size)
test_size  = total_size - train_size - val_size

data_train, data_val, data_test = random_split(full_dataset, [train_size, val_size, test_size])

del full_dataset

train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8)
val_loader   = DataLoader(data_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
test_loader  = DataLoader(data_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# Models
model_r1_to_r2 = ActionMapperMLP(ACTION_DIM_R1, ACTION_DIM_R2, HIDDEN_DIM)
model_r2_to_r1 = ActionMapperMLP(ACTION_DIM_R2, ACTION_DIM_R1, HIDDEN_DIM)

# Loss and optimizers
mse_loss = nn.MSELoss()

optimizer_r1_to_r2 = torch.optim.Adam(model_r1_to_r2.parameters(), lr=LEARNING_RATE)
optimizer_r2_to_r1 = torch.optim.Adam(model_r2_to_r1.parameters(), lr=LEARNING_RATE)

# History
history = {
    "train_r1_to_r2": [],
    "val_r1_to_r2":   [],
    "train_r2_to_r1": [],
    "val_r2_to_r1":   [],
}

# ─────────────────────────────────────────────
# CSV log
# ─────────────────────────────────────────────
LOG_PATH   = os.path.join(LOG_DIR, "training_log.csv")
CSV_FIELDS = ["epoch",
              "train_r1_to_r2", "val_r1_to_r2",
              "train_r2_to_r1", "val_r2_to_r1"]

log_file   = open(LOG_PATH, "w", newline="")
log_writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
log_writer.writeheader()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def evaluate_val(loader):
    """Compute mean MSE over the full validation set."""
    sum_r1_to_r2 = 0.0
    sum_r2_to_r1 = 0.0
    n = 0
    for action_r1, action_r2 in loader:
        sum_r1_to_r2 += mse_loss(model_r1_to_r2(action_r1), action_r2).item()
        sum_r2_to_r1 += mse_loss(model_r2_to_r1(action_r2), action_r1).item()
        n += 1
    return sum_r1_to_r2 / n, sum_r2_to_r1 / n


def save_plot(history, path):
    """Save train/val loss curves for both directions."""
    epochs = range(1, len(history["train_r1_to_r2"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

    # r1 → r2
    ax1.plot(epochs, history["train_r1_to_r2"], label="train", color="steelblue",  linewidth=1.8)
    ax1.plot(epochs, history["val_r1_to_r2"],   label="val",   color="darkorange", linewidth=1.8, linestyle="--")
    ax1.set_title("MSE  r1 → r2")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.legend()
    ax1.grid(True, alpha=0.35)

    # r2 → r1
    ax2.plot(epochs, history["train_r2_to_r1"], label="train", color="steelblue",  linewidth=1.8)
    ax2.plot(epochs, history["val_r2_to_r1"],   label="val",   color="darkorange", linewidth=1.8, linestyle="--")
    ax2.set_title("MSE  r2 → r1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.legend()
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved → {}".format(path))

# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
if __name__ == "__main__":

    pbar = tqdm(range(NUM_EPOCHS), desc="Training Actions Mappers")

    for epoch in pbar:

        # ── Train ─────────────────────────────
        model_r1_to_r2.train()
        model_r2_to_r1.train()

        train_sum_r1_to_r2 = 0.0
        train_sum_r2_to_r1 = 0.0

        for action_r1, action_r2 in train_loader:

            # r1 → r2
            batch_loss = mse_loss(model_r1_to_r2(action_r1), action_r2)
            optimizer_r1_to_r2.zero_grad()
            batch_loss.backward()
            optimizer_r1_to_r2.step()
            train_sum_r1_to_r2 += batch_loss.item()

            # r2 → r1
            batch_loss = mse_loss(model_r2_to_r1(action_r2), action_r1)
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

        with torch.no_grad():
            val_r1_to_r2, val_r2_to_r1 = evaluate_val(val_loader)

        history["val_r1_to_r2"].append(val_r1_to_r2)
        history["val_r2_to_r1"].append(val_r2_to_r1)

        # ── Console log ───────────────────────
        '''
        print(
            "Epoch [{:>3}/{}]  "
            "train({:.6f} / {:.6f})  "
            "val({:.6f} / {:.6f})".format(
                epoch + 1, NUM_EPOCHS,
                train_r1_to_r2, train_r2_to_r1,
                val_r1_to_r2,   val_r2_to_r1,
            )
        )
        '''
        pbar.set_postfix({"Epoch": f"[{epoch+1:>3}/{NUM_EPOCHS}]", 
                          "train": f"({train_r1_to_r2:.6f}/{train_r2_to_r1:.6f})", 
                          "val": f"({val_r1_to_r2:.6f}/{val_r2_to_r1:.6f})"
                          })


        # ── CSV row ───────────────────────────
        log_writer.writerow({
            "epoch":          epoch + 1,
            "train_r1_to_r2": train_r1_to_r2,
            "val_r1_to_r2":   val_r1_to_r2,
            "train_r2_to_r1": train_r2_to_r1,
            "val_r2_to_r1":   val_r2_to_r1,
        })
        log_file.flush()

    # ── Qualitative sample ────────────────
    with torch.no_grad():
        idx      = np.random.randint(0, len(data_val))
        a_r1, a_r2 = data_val[idx]

        pred_r2 = model_r1_to_r2(torch.from_numpy(a_r1).unsqueeze(0)).squeeze(0)
        pred_r1 = model_r2_to_r1(torch.from_numpy(a_r2).unsqueeze(0)).squeeze(0)

        print("─" * 100)
        print("  [r1→r2] target : {}".format([round(v, 4) for v in a_r2.tolist()]))
        print("          pred   : {}".format([round(v, 4) for v in pred_r2.tolist()]))
        print("  [r2→r1] target : {}".format([round(v, 4) for v in a_r1.tolist()]))
        print("          pred   : {}".format([round(v, 4) for v in pred_r1.tolist()]))
        print("  Pred std r1→r2 : {:.4f}  Target std: {:.4f}".format(
            pred_r2.std().item(), a_r2.std().item()))

    print("─" * 100)

    # ─────────────────────────────────────────
    # Save models
    # ─────────────────────────────────────────
    log_file.close()

    for name, model in [("action_mapper_r1_to_r2", model_r1_to_r2),
                         ("action_mapper_r2_to_r1", model_r2_to_r1)]:
        path = os.path.join(MODEL_DIR, "{}.pt".format(name))
        torch.save(model.state_dict(), path)
        print("Model saved → {}".format(path))

    # ─────────────────────────────────────────
    # Save plot
    # ─────────────────────────────────────────
    plot_path = os.path.join(PLOT_DIR, "loss_curves.png")
    save_plot(history, plot_path)

    print("\nDone. All outputs in: {}".format(RUN_DIR))
    print("  models/ — r1_to_r2.pt, r2_to_r1.pt")
    print("  logs/   — training_log.csv")
    print("  plots/  — loss_curves.png")