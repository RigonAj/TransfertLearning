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

from .mapper_models import StateMapperMLP, ActionMapperMLP

torch.set_num_threads(4)


# =========================================================
# Utilitaire : position cartésienne depuis un état normalisé
# =========================================================

def ee_from_state(state: np.ndarray, dof: int) -> np.ndarray:
    """
    Calcule la position de l'effecteur à partir d'un état normalisé.
    Les états stockés sont θ/π → dénormalisation par π.
    """

    if dof == 2:
        t1 = state[0] * np.pi
        t2 = state[1] * np.pi

        l1, l2 = 1.5, 1.5

        x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
        y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)

    else:
        t1 = state[0] * np.pi
        t2 = state[1] * np.pi
        t3 = state[2] * np.pi

        l1 = l2 = l3 = 1.0

        x = (
            l1 * np.cos(t1)
            + l2 * np.cos(t1 + t2)
            + l3 * np.cos(t1 + t2 + t3)
        )

        y = (
            l1 * np.sin(t1)
            + l2 * np.sin(t1 + t2)
            + l3 * np.sin(t1 + t2 + t3)
        )

    return np.array([x, y], dtype=np.float32)


# =========================================================
# Validation des données
# =========================================================

def validate_trajectories(trajectories: Dict):

    checks = {
        "segments_2dof":
            trajectories["segments_2dof"],

        "segments_3dof":
            trajectories["segments_3dof"],

        "segments_actions_2dof":
            trajectories["segments_actions_2dof"],

        "segments_actions_3dof":
            trajectories["segments_actions_3dof"],
    }

    for name, arr in checks.items():

        n_nan = np.sum(np.isnan(arr))
        n_inf = np.sum(np.isinf(arr))

        if n_nan > 0 or n_inf > 0:
            raise ValueError(
                f"[DATA] {name} contient "
                f"{n_nan} NaN et {n_inf} Inf"
            )

    print("[DATA] Validation OK")


# =========================================================
# Dataset
# =========================================================

class SegmentDataset(Dataset):

    def __init__(self, s_in, s_out, a_in, a_out):

        self.s_in = torch.from_numpy(s_in).float()
        self.s_out = torch.from_numpy(s_out).float()

        self.a_in = torch.from_numpy(a_in).float()
        self.a_out = torch.from_numpy(a_out).float()

    def __len__(self):
        return len(self.s_in)

    def __getitem__(self, idx):

        return (
            self.s_in[idx],
            self.s_out[idx],
            self.a_in[idx],
            self.a_out[idx],
        )


# =========================================================
# Transfer mapper
# =========================================================

class TransferMapper:

    def __init__(
        self,
        in_state_dim,
        out_state_dim,
        in_act_dim,
        out_act_dim,
        device="cpu",
        log_dir=None,
        label="transfer",
    ):

        self.device = device
        self.label = label

        self.writer = (
            SummaryWriter(log_dir)
            if log_dir
            else None
        )

        self.state_mapper = StateMapperMLP(
            in_state_dim,
            out_state_dim
        ).to(device)

        self.action_mapper = ActionMapperMLP(
            in_state_dim,
            in_act_dim,
            out_act_dim
        ).to(device)

        self.opt_state = optim.Adam(
            self.state_mapper.parameters(),
            lr=3e-4,
            weight_decay=1e-5
        )

        self.opt_action = optim.Adam(
            self.action_mapper.parameters(),
            lr=3e-4,
            weight_decay=1e-5
        )

        self.criterion = nn.MSELoss()

        self.in_state_dim = in_state_dim
        self.out_state_dim = out_state_dim

        self.dof_in = (
            3 if in_state_dim == 8
            else 2
        )

        self.dof_out = (
            3 if out_state_dim == 8
            else 2
        )

    # =====================================================
    # TRAIN
    # =====================================================

    def train(
        self,
        s_in_all,
        s_out_all,
        a_in_all,
        a_out_all,
        epochs=1000,
        batch_size=32,
        patience=50,
    ):

        N = s_in_all.shape[0]

        print(
            f"\n=== {self.label} | "
            f"{N} segments ==="
        )

        rng = np.random.RandomState(0)

        idx = rng.permutation(N)

        n_val = max(
            1,
            int(0.1 * N)
        )

        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        full_ds = SegmentDataset(
            s_in_all,
            s_out_all,
            a_in_all,
            a_out_all,
        )

        train_dl = DataLoader(
            torch.utils.data.Subset(
                full_ds,
                train_idx
            ),
            batch_size=batch_size,
            shuffle=True
        )

        val_dl = DataLoader(
            torch.utils.data.Subset(
                full_ds,
                val_idx
            ),
            batch_size=batch_size,
            shuffle=False
        )

        best_val = np.inf

        wait = 0

        best_state = None
        best_action = None

        pbar = tqdm(
            range(epochs),
            desc=self.label
        )

        for epoch in pbar:

            self.state_mapper.train()
            self.action_mapper.train()

            tr_s = 0.0
            tr_a = 0.0

            for s_in, s_out, a_in, a_out in train_dl:

                B, L, _ = s_in.shape

                s_in_f = (
                    s_in.reshape(B * L, -1)
                    .to(self.device)
                )

                s_out_f = (
                    s_out.reshape(B * L, -1)
                    .to(self.device)
                )

                a_in_f = (
                    a_in.reshape(B * L, -1)
                    .to(self.device)
                )

                a_out_f = (
                    a_out.reshape(B * L, -1)
                    .to(self.device)
                )

                self.opt_state.zero_grad()

                pred_s = self.state_mapper(
                    s_in_f
                )

                loss_s = self.criterion(
                    pred_s,
                    s_out_f
                )

                loss_s.backward()

                nn.utils.clip_grad_norm_(
                    self.state_mapper.parameters(),
                    1.0
                )

                self.opt_state.step()

                self.opt_action.zero_grad()

                pred_a = self.action_mapper(
                    s_in_f,
                    a_in_f
                )

                loss_a = self.criterion(
                    pred_a,
                    a_out_f
                )

                loss_a.backward()

                nn.utils.clip_grad_norm_(
                    self.action_mapper.parameters(),
                    1.0
                )

                self.opt_action.step()

                tr_s += loss_s.item()
                tr_a += loss_a.item()

            tr_s /= len(train_dl)
            tr_a /= len(train_dl)

            # ================= VAL =================

            self.state_mapper.eval()
            self.action_mapper.eval()

            val_s = 0.0
            val_a = 0.0

            with torch.no_grad():

                for s_in, s_out, a_in, a_out in val_dl:

                    B, L, _ = s_in.shape

                    s_in_f = (
                        s_in.reshape(B * L, -1)
                        .to(self.device)
                    )

                    s_out_f = (
                        s_out.reshape(B * L, -1)
                        .to(self.device)
                    )

                    a_in_f = (
                        a_in.reshape(B * L, -1)
                        .to(self.device)
                    )

                    a_out_f = (
                        a_out.reshape(B * L, -1)
                        .to(self.device)
                    )

                    val_s += self.criterion(
                        self.state_mapper(
                            s_in_f
                        ),
                        s_out_f
                    ).item()

                    val_a += self.criterion(
                        self.action_mapper(
                            s_in_f,
                            a_in_f
                        ),
                        a_out_f
                    ).item()

            val_s /= len(val_dl)
            val_a /= len(val_dl)

            total_val = val_s + val_a

            fidelity = self._fidelity(
                s_in_all[val_idx],
                s_out_all[val_idx],
                a_in_all[val_idx],
                a_out_all[val_idx],
                n_samples=min(
                    200,
                    n_val
                )
            )

            pbar.set_postfix(
                tr_s=f"{tr_s:.5f}",
                tr_a=f"{tr_a:.5f}",
                v_s=f"{val_s:.5f}",
                v_a=f"{val_a:.5f}",
                ee=f"{fidelity['ee_error']:.5f}"
            )

            if self.writer is not None:
                self.writer.add_scalar("Loss/train_state", tr_s, epoch)
                self.writer.add_scalar("Loss/train_action", tr_a, epoch)
                self.writer.add_scalar("Loss/val_state", val_s, epoch)
                self.writer.add_scalar("Loss/val_action", val_a, epoch)
                self.writer.add_scalar("Fidelity/ee_error", fidelity["ee_error"], epoch)
                self.writer.flush()

            if total_val < best_val - 1e-6:

                best_val = total_val

                wait = 0

                best_state = {
                    k: v.clone()
                    for k, v in
                    self.state_mapper
                    .state_dict()
                    .items()
                }

                best_action = {
                    k: v.clone()
                    for k, v in
                    self.action_mapper
                    .state_dict()
                    .items()
                }

            else:

                wait += 1

                if wait >= patience:

                    print(
                        f"Early stop "
                        f"epoch {epoch}"
                    )

                    break

        if best_state is not None:

            self.state_mapper.load_state_dict(
                best_state
            )

            self.action_mapper.load_state_dict(
                best_action
            )

    # =====================================================
    # FIDELITY
    # =====================================================

    @torch.no_grad()
    def _fidelity(
        self,
        s_in_all,
        s_out_all,
        a_in_all,
        a_out_all,
        n_samples=200
    ):

        n = min(
            n_samples,
            len(s_in_all)
        )

        idx = np.random.choice(
            len(s_in_all),
            n,
            replace=False
        )

        action_mse = 0.0
        ee_error = 0.0
        full_mse = 0.0

        self.state_mapper.eval()
        self.action_mapper.eval()

        for i in idx:

            s_in = torch.from_numpy(
                s_in_all[i]
            ).float().to(self.device)

            a_in = torch.from_numpy(
                a_in_all[i]
            ).float().to(self.device)

            s_out_gt = torch.from_numpy(
                s_out_all[i]
            ).float().to(self.device)

            a_out_gt = torch.from_numpy(
                a_out_all[i]
            ).float().to(self.device)

            s_out_pred = self.state_mapper(
                s_in
            )

            a_out_pred = self.action_mapper(
                s_in,
                a_in
            )

            action_mse += self.criterion(
                a_out_pred,
                a_out_gt
            ).item()

            L = s_in.shape[0]

            seg_ee = 0.0

            for t in range(L):

                ee_pred = ee_from_state(
                    s_out_pred[t]
                    .cpu()
                    .numpy(),
                    self.dof_out
                )

                ee_real = ee_from_state(
                    s_out_gt[t]
                    .cpu()
                    .numpy(),
                    self.dof_out
                )

                seg_ee += (
                    np.linalg.norm(
                        ee_pred
                        - ee_real
                    ) ** 2
                )

            ee_error += seg_ee / L

            pred_cat = torch.cat(
                [
                    s_out_pred,
                    a_out_pred
                ],
                dim=-1
            )

            gt_cat = torch.cat(
                [
                    s_out_gt,
                    a_out_gt
                ],
                dim=-1
            )

            full_mse += self.criterion(
                pred_cat,
                gt_cat
            ).item()

        return {
            "action_mse":
                action_mse / n,

            "ee_error":
                ee_error / n,

            "full_mse":
                full_mse / n,
        }

    # =====================================================
    # SAVE
    # =====================================================

    def save(self, path):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            {
                "state_mapper":
                    self.state_mapper.state_dict(),

                "action_mapper":
                    self.action_mapper.state_dict(),
            },
            path
        )

        print(
            f"Saved {self.label}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_dir = Path(
        "./data/DIRECT"
    )

    with open(
        data_dir /
        "trajectories_aligned_ss.pkl",
        "rb"
    ) as f:

        trajectories = pickle.load(f)

    validate_trajectories(
        trajectories
    )

    meta = trajectories[
        "metadata"
    ]

    print(
        f"Loaded "
        f"{meta['n_segments']} "
        f"segments"
    )

    s2 = trajectories[
        "segments_2dof"
    ]

    s3 = trajectories[
        "segments_3dof"
    ]

    a2 = trajectories[
        "segments_actions_2dof"
    ]

    a3 = trajectories[
        "segments_actions_3dof"
    ]

    log_dir = (
        data_dir /
        "mappers_logs"
    )

    t23 = TransferMapper(
        8,
        6,
        2,
        3,
        device=device,
        log_dir=str(
            log_dir /
            "2to3"
        ),
        label="2→3"
    )

    t23.train(
        s3,
        s2,
        a2,
        a3,
        epochs=1200
    )

    t23.save(
        data_dir /
        "transfer_2to3_seq.pt"
    )

    t32 = TransferMapper(
        6,
        8,
        3,
        2,
        device=device,
        log_dir=str(
            log_dir /
            "3to2"
        ),
        label="3→2"
    )

    t32.train(
        s2,
        s3,
        a3,
        a2,
        epochs=1200
    )

    t32.save(
        data_dir /
        "transfer_3to2_seq.pt"
    )

    print(
        "\n=== TERMINE ==="
    )


if __name__ == "__main__":
    main()
