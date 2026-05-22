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
import os
import platform

from .mapper_models import (
    StateMapperMLP,
    ActionMapperMLP,
    arm_effector_fields_torch,
    rebuild_arm_state_with_fk,
)

# global torch thread count is configured per-run (see `train()`)


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

        # normalize device to torch.device
        if not isinstance(device, torch.device):
            device = torch.device(device)

        self.device = device
        self.label = label

        # AMP scaler if using CUDA
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.writer = (
            SummaryWriter(log_dir)
            if log_dir
            else None
        )

        self.state_mapper = StateMapperMLP(
            in_state_dim,
            out_state_dim
        ).to(self.device)

        self.action_mapper = ActionMapperMLP(
            in_state_dim,
            in_act_dim,
            out_act_dim
        ).to(self.device)

        # Optionally try to JIT-compile models with torch.compile (PyTorch 2.x)
        # Prefer compiling only when the Triton backend is available and
        # known-working; torch.compile may defer work until first call and
        # then raise runtime errors (Inductor/Triton missing). To avoid that
        # surprising failure, check for Triton first and catch compile errors.
        # Only compile models if explicitly enabled via environment variable.
        # Automatic compilation can cause runtime failures (Inductor/Triton
        # backend missing or incompatible). To enable: set
        # `TORCH_COMPILE=1` in the environment.
        if hasattr(torch, "compile") and os.environ.get("TORCH_COMPILE", "0") in ("1", "true", "True"):
            try:
                # Prefer to check for Triton presence; if it's not available,
                # skip compilation to avoid Inductor runtime errors.
                import triton  # type: ignore
                triton_ok = True
            except Exception:
                triton_ok = False

            if triton_ok:
                try:
                    self.state_mapper = torch.compile(self.state_mapper)
                    self.action_mapper = torch.compile(self.action_mapper)
                    print("[INFO] Models compiled with torch.compile")
                except Exception as e:
                    print(f"[WARN] torch.compile failed at compile-time: {e}")
            else:
                print("[INFO] torch.compile skipped: Triton not available or disabled")
        else:
            print("[INFO] torch.compile disabled (set TORCH_COMPILE=1 to enable)")

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

    def _state_loss(self, raw_pred_s, s_out_f, s_in_f):
        """
        Train the state mapper as a task-agnostic arm mapper.

        Angles/velocities are still guided by the paired source trajectory, but
        the end-effector is forced toward the real input robot end-effector.
        This keeps reaching-trained mappers meaningful for later tasks whose
        observations depend on the current Cartesian arm state.
        """
        pred_s = rebuild_arm_state_with_fk(raw_pred_s)

        eff_start = 4 if self.out_state_dim == 6 else 6
        loss_pose = self.criterion(
            pred_s[..., :eff_start],
            s_out_f[..., :eff_start],
        )

        ref_eff = arm_effector_fields_torch(s_in_f)
        pred_eff = arm_effector_fields_torch(pred_s)
        raw_eff = arm_effector_fields_torch(raw_pred_s)

        loss_ref_ee = self.criterion(pred_eff, ref_eff)
        loss_raw_ee = self.criterion(raw_eff, ref_eff)

        loss = loss_pose + 5.0 * loss_ref_ee + 0.25 * loss_raw_ee
        return loss, pred_s

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
        batch_size=1024,
        patience=50,
        num_workers=None,
        prefetch_factor=None,
        pin_memory=None,
        accumulation_steps=1,
        dataset_half=False,
        worker_cap=12,
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

        # GPU performance settings
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # compute DataLoader workers and torch threads to avoid oversubscription
        n_cpus = os.cpu_count() or 1
        cap = max(1, min(worker_cap, n_cpus - 1))

        if num_workers is not None:
            n_workers = int(num_workers)
        else:
            if platform.system() == "Windows":
                # Windows: spawn overhead higher — use roughly half logical CPUs
                n_workers = max(0, min(max(1, n_cpus // 2), cap))
            else:
                # Unix-like: use CPUs-1 but cap to `cap`
                n_workers = max(0, min(n_cpus - 1, cap))

        # threads per process (intra-op) to avoid too many threads when using workers
        threads_per_worker = max(1, n_cpus // (n_workers + 1))

        # set environment vars for BLAS/OpenMP libraries
        os.environ.setdefault('OMP_NUM_THREADS', str(threads_per_worker))
        os.environ.setdefault('OPENBLAS_NUM_THREADS', str(threads_per_worker))
        os.environ.setdefault('MKL_NUM_THREADS', str(threads_per_worker))

        try:
            torch.set_num_threads(threads_per_worker)
        except Exception:
            pass

        try:
            # setting interop threads may fail if parallel work already started
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        if pin_memory is None:
            pin_memory = True if self.device.type == "cuda" else False

        pf = prefetch_factor if prefetch_factor is not None else 4

        # If no workers are used, prefetch_factor must be None (DataLoader validation)
        if n_workers == 0:
            if pf is not None:
                print("[WARN] num_workers==0, overriding prefetch_factor -> None to avoid DataLoader error")
            pf = None

        print(f"[INFO] DataLoader num_workers={n_workers}, pin_memory={pin_memory}, threads_per_worker={threads_per_worker}, n_cpus={n_cpus}, prefetch_factor={pf}")

        # Optionally convert dataset to half precision to reduce transfer size
        if dataset_half and self.device.type == "cuda" and self.use_amp:
            try:
                full_ds.s_in = full_ds.s_in.half()
                full_ds.s_out = full_ds.s_out.half()
                full_ds.a_in = full_ds.a_in.half()
                full_ds.a_out = full_ds.a_out.half()
            except Exception:
                pass

        # Pre-pin dataset tensors (speeds up cuda transfers) when possible
        if self.device.type == "cuda":
            try:
                for name in ('s_in', 's_out', 'a_in', 'a_out'):
                    t = getattr(full_ds, name)
                    if isinstance(t, torch.Tensor) and not t.is_pinned():
                        try:
                            setattr(full_ds, name, t.pin_memory())
                        except Exception:
                            pass
            except Exception:
                pass

        train_dl = DataLoader(
            torch.utils.data.Subset(full_ds, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=pin_memory,
            persistent_workers=(n_workers > 0),
            prefetch_factor=pf,
        )

        val_dl = DataLoader(
            torch.utils.data.Subset(full_ds, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=pin_memory,
            persistent_workers=(n_workers > 0),
            prefetch_factor=pf,
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


            # initialize grads for accumulation
            self.opt_state.zero_grad()
            self.opt_action.zero_grad()

            for batch_idx, (s_in, s_out, a_in, a_out) in enumerate(train_dl):

                B, L, _ = s_in.shape

                s_in_f = s_in.reshape(B * L, -1).to(self.device, non_blocking=True)
                s_out_f = s_out.reshape(B * L, -1).to(self.device, non_blocking=True)
                a_in_f = a_in.reshape(B * L, -1).to(self.device, non_blocking=True)
                a_out_f = a_out.reshape(B * L, -1).to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        raw_pred_s = self.state_mapper(s_in_f)
                        loss_s, _ = self._state_loss(raw_pred_s, s_out_f, s_in_f)
                        pred_a = self.action_mapper(s_in_f, a_in_f)
                        loss_a = self.criterion(pred_a, a_out_f)

                    total_loss = loss_s + loss_a
                    loss_to_back = total_loss / accumulation_steps
                    self.scaler.scale(loss_to_back).backward()

                else:
                    raw_pred_s = self.state_mapper(s_in_f)
                    loss_s, _ = self._state_loss(raw_pred_s, s_out_f, s_in_f)
                    pred_a = self.action_mapper(s_in_f, a_in_f)
                    loss_a = self.criterion(pred_a, a_out_f)

                    total_loss = loss_s + loss_a
                    (total_loss / accumulation_steps).backward()

                tr_s += loss_s.item()
                tr_a += loss_a.item()

                # step when we've accumulated enough or at last batch
                is_last = (batch_idx == len(train_dl) - 1)
                if ((batch_idx + 1) % accumulation_steps == 0) or is_last:
                    if self.use_amp:
                        # unscale before clipping
                        try:
                            self.scaler.unscale_(self.opt_state)
                        except Exception:
                            pass
                        nn.utils.clip_grad_norm_(self.state_mapper.parameters(), 1.0)
                        try:
                            self.scaler.unscale_(self.opt_action)
                        except Exception:
                            pass
                        nn.utils.clip_grad_norm_(self.action_mapper.parameters(), 1.0)

                        self.scaler.step(self.opt_state)
                        self.scaler.step(self.opt_action)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.state_mapper.parameters(), 1.0)
                        nn.utils.clip_grad_norm_(self.action_mapper.parameters(), 1.0)

                        self.opt_state.step()
                        self.opt_action.step()

                    self.opt_state.zero_grad()
                    self.opt_action.zero_grad()

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

                    s_in_f = s_in.reshape(B * L, -1).to(self.device, non_blocking=True)
                    s_out_f = s_out.reshape(B * L, -1).to(self.device, non_blocking=True)
                    a_in_f = a_in.reshape(B * L, -1).to(self.device, non_blocking=True)
                    a_out_f = a_out.reshape(B * L, -1).to(self.device, non_blocking=True)

                    if self.use_amp:
                        with torch.amp.autocast("cuda"):
                            raw_pred_s = self.state_mapper(s_in_f)
                            loss_s, _ = self._state_loss(raw_pred_s, s_out_f, s_in_f)

                            val_s += loss_s.item()
                            val_a += self.criterion(self.action_mapper(s_in_f, a_in_f), a_out_f).item()
                    else:
                        raw_pred_s = self.state_mapper(s_in_f)
                        loss_s, _ = self._state_loss(raw_pred_s, s_out_f, s_in_f)

                        val_s += loss_s.item()
                        val_a += self.criterion(self.action_mapper(s_in_f, a_in_f), a_out_f).item()

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

        n = min(n_samples, len(s_in_all))

        if n == 0:
            return {"action_mse": 0.0, "ee_error": 0.0, "full_mse": 0.0}

        idx = np.random.choice(len(s_in_all), n, replace=False)

        # Move the selected validation samples to device in a single batch
        s_in_batch = torch.from_numpy(s_in_all[idx]).float().to(self.device, non_blocking=True)
        a_in_batch = torch.from_numpy(a_in_all[idx]).float().to(self.device, non_blocking=True)
        s_out_gt_batch = torch.from_numpy(s_out_all[idx]).float().to(self.device, non_blocking=True)
        a_out_gt_batch = torch.from_numpy(a_out_all[idx]).float().to(self.device, non_blocking=True)

        B = s_in_batch.shape[0]
        L = s_in_batch.shape[1]

        # Flatten time dimension for the model forward passes
        s_in_f = s_in_batch.reshape(B * L, -1)
        a_in_f = a_in_batch.reshape(B * L, -1)
        s_out_gt_f = s_out_gt_batch.reshape(B * L, -1)
        a_out_gt_f = a_out_gt_batch.reshape(B * L, -1)

        self.state_mapper.eval()
        self.action_mapper.eval()

        # Forward all at once
        s_out_pred_f = rebuild_arm_state_with_fk(self.state_mapper(s_in_f))
        a_out_pred_f = self.action_mapper(s_in_f, a_in_f)

        # reshape back to (B, L, ...)
        s_out_pred = s_out_pred_f.reshape(B, L, -1)
        a_out_pred = a_out_pred_f.reshape(B, L, -1)

        # action mse averaged over all time steps and samples
        action_mse = self.criterion(a_out_pred_f, a_out_gt_f).item()

        # full mse averaged over all time steps and samples
        pred_cat = torch.cat([s_out_pred_f, a_out_pred_f], dim=-1)
        gt_cat = torch.cat([s_out_gt_f, a_out_gt_f], dim=-1)
        full_mse = self.criterion(pred_cat, gt_cat).item()

        # ee error: use the end-effector fields (already present in states) and compute MSE
        ee_pred = arm_effector_fields_torch(s_out_pred)
        ee_real = arm_effector_fields_torch(s_in_batch)

        # per-time squared error, mean over time and samples
        ee_error = (ee_pred - ee_real).pow(2).sum(dim=-1).mean().item()

        return {
            "action_mse": action_mse,
            "ee_error": ee_error,
            "full_mse": full_mse,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")

    if device.type == "cuda":
        print(f"[INFO] torch.version.cuda = {torch.version.cuda}; device_count = {torch.cuda.device_count()}")
        # Enable TF32 and higher float32 matmul precision for faster matmuls on Ampere+ GPUs
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

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

    # If temporal variants exist, build combined inputs by concatenating
    # spatial + temporal features along the last axis. Keep targets (outputs)
    # in their original spatial form to preserve downstream expectations.
    if (
        "segments_2dof_temporal" in trajectories
        and "segments_3dof_temporal" in trajectories
        and "segments_actions_2dof_temporal" in trajectories
        and "segments_actions_3dof_temporal" in trajectories
    ):
        print("[INFO] Temporal variants detected — creating combined inputs (spatial+temporal)")

        s2_t = trajectories["segments_2dof_temporal"]
        s3_t = trajectories["segments_3dof_temporal"]
        a2_t = trajectories["segments_actions_2dof_temporal"]
        a3_t = trajectories["segments_actions_3dof_temporal"]

        # basic shape checks
        if s2.shape != s2_t.shape or s3.shape != s3_t.shape:
            raise ValueError("Spatial and temporal state variants have mismatched shapes")
        if a2.shape != a2_t.shape or a3.shape != a3_t.shape:
            raise ValueError("Spatial and temporal action variants have mismatched shapes")

        # concatenate along feature dim
        s2_comb = np.concatenate([s2, s2_t], axis=-1)
        s3_comb = np.concatenate([s3, s3_t], axis=-1)
        a2_comb = np.concatenate([a2, a2_t], axis=-1)
        a3_comb = np.concatenate([a3, a3_t], axis=-1)

        # Use combined inputs as the mapper inputs; keep outputs original spatial
        s2_in = s2_comb
        s3_in = s3_comb
        a2_in = a2_comb
        a3_in = a3  # keep a3 as original (out action remains original dims)

    else:
        print("[INFO] No temporal variants found — using spatial-only inputs")
        s2_in = s2
        s3_in = s3
        a2_in = a2
        a3_in = a3

    log_dir = (
        data_dir /
        "mappers_logs"
    )

    # Configure and train mapper 2→3 (state: 3dof -> 2dof, action: 2->3)
    t23_in_state_dim = s3_in.shape[-1]
    t23_out_state_dim = s2.shape[-1]
    t23_in_act_dim = a2_in.shape[-1]
    t23_out_act_dim = a3.shape[-1]

    t23 = TransferMapper(
        t23_in_state_dim,
        t23_out_state_dim,
        t23_in_act_dim,
        t23_out_act_dim,
        device=device,
        log_dir=str(
            log_dir /
            "2to3"
        ),
        label="2→3_combined" if s3_in.shape[-1] != s3.shape[-1] else "2→3",
    )

    t23.train(
        s3_in,
        s2,
        a2_in,
        a3_in,
        epochs=1200,
        batch_size=256,
        num_workers=0,
        prefetch_factor=4,
        dataset_half=True,
    )

    t23.save(
        data_dir /
        "transfer_2to3_seq.pt"
    )

    # Configure and train mapper 3→2 (state: 2dof -> 3dof, action: 3->2)
    t32_in_state_dim = s2_in.shape[-1]
    t32_out_state_dim = s3.shape[-1]
    t32_in_act_dim = a3_in.shape[-1]
    t32_out_act_dim = a2.shape[-1]

    t32 = TransferMapper(
        t32_in_state_dim,
        t32_out_state_dim,
        t32_in_act_dim,
        t32_out_act_dim,
        device=device,
        log_dir=str(
            log_dir /
            "3to2"
        ),
        label="3→2_combined" if s2_in.shape[-1] != s2.shape[-1] else "3→2",
    )

    t32.train(
        s2_in,
        s3,
        a3_in,
        a2,
        epochs=1200,
        batch_size=256,
        num_workers=0,
        prefetch_factor=4,
        dataset_half=True,
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
