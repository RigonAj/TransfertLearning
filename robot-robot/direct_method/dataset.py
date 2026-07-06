import numpy as np
from torch.utils.data import Dataset


class TrajectoriesTrainingDataset(Dataset):
    """
    Each trajectory row is assumed to be laid out as:
        [theta_1 ... theta_nq | dtheta_1 ... dtheta_nq | eff_x  eff_y]
    i.e. nq joint positions, nq joint velocities, then 2 end-effector
    Cartesian coordinates  ->  total columns = 2*nq + 2.

    NOTE: nq_r1 / nq_r2 (not dim_R1/dim_R2) now drive every slice, so this
    works correctly whether the trailing eff_x/eff_y columns are included
    in the mapper's I/O dimension or not.
    """
    def __init__(self, trajectories_r1, trajectories_r2,
                 nq_r1, nq_r2,
                 omega_max=2.0, max_reach=3.0):
        self.nq_r1 = nq_r1
        self.nq_r2 = nq_r2

        dim_r1 = 2 * nq_r1 + 2
        dim_r2 = 2 * nq_r2 + 2

        self.trajectories_r1 = np.loadtxt(trajectories_r1, dtype=np.float32)[:, :dim_r1]
        self.trajectories_r2 = np.loadtxt(trajectories_r2, dtype=np.float32)[:, :dim_r2]

        # ── Validity mask (raw angular velocities must lie within +/- omega_max) ──
        vel_r1 = self.trajectories_r1[:, nq_r1:2 * nq_r1]
        vel_r2 = self.trajectories_r2[:, nq_r2:2 * nq_r2]

        valid_mask_r1 = np.all((vel_r1 >= -omega_max) & (vel_r1 <= omega_max), axis=1)
        valid_mask_r2 = np.all((vel_r2 >= -omega_max) & (vel_r2 <= omega_max), axis=1)
        valid_mask = valid_mask_r1 & valid_mask_r2

        self.trajectories_r1 = self.trajectories_r1[valid_mask]
        self.trajectories_r2 = self.trajectories_r2[valid_mask]

        # ── Normalisation ──────────────────────────────────────────────────
        # positions   -> / pi
        self.trajectories_r1[:, :nq_r1] /= np.pi
        self.trajectories_r2[:, :nq_r2] /= np.pi
        # velocities  -> / omega_max
        self.trajectories_r1[:, nq_r1:2 * nq_r1] /= omega_max
        self.trajectories_r2[:, nq_r2:2 * nq_r2] /= omega_max
        # end-effector -> / max_reach   (only meaningful if dim_R == 2*nq + 2)
        if self.trajectories_r1.shape[1] > 2 * nq_r1:
            self.trajectories_r1[:, 2 * nq_r1:] /= max_reach
        if self.trajectories_r2.shape[1] > 2 * nq_r2:
            self.trajectories_r2[:, 2 * nq_r2:] /= max_reach

    def __len__(self):
        return len(self.trajectories_r1)

    def __getitem__(self, idx):
        return self.trajectories_r1[idx], self.trajectories_r2[idx]


class TrajectoriesTrainingActionDataset(Dataset):
    """
    Extracts the VELOCITY block [dtheta_1 ... dtheta_nq] from each
    trajectory file (NOT the leading joint-ANGLE columns, which is what
    the previous implementation accidentally did via `[:, :action_dim]`).

    This is what the policy's `action` actually looks like at runtime in
    the env (a normalised commanded velocity in [-1, 1]), so the action
    mapper has to be trained on the same quantity.
    """
    def __init__(self, trajectories_r1, trajectories_r2,
                 nq_r1, nq_r2, omega_max=2.0):
        dim_r1 = 2 * nq_r1 + 2
        dim_r2 = 2 * nq_r2 + 2

        raw_r1 = np.loadtxt(trajectories_r1, dtype=np.float32)[:, :dim_r1]
        raw_r2 = np.loadtxt(trajectories_r2, dtype=np.float32)[:, :dim_r2]

        vel_r1 = raw_r1[:, nq_r1:2 * nq_r1]
        vel_r2 = raw_r2[:, nq_r2:2 * nq_r2]

        # Same validity criterion as the state dataset, applied here too so
        # the action mapper isn't trained on out-of-range velocity outliers.
        valid_mask_r1 = np.all((vel_r1 >= -omega_max) & (vel_r1 <= omega_max), axis=1)
        valid_mask_r2 = np.all((vel_r2 >= -omega_max) & (vel_r2 <= omega_max), axis=1)
        valid_mask = valid_mask_r1 & valid_mask_r2

        self.trajectories_r1 = vel_r1[valid_mask] / omega_max
        self.trajectories_r2 = vel_r2[valid_mask] / omega_max

    def __len__(self):
        return len(self.trajectories_r1)

    def __getitem__(self, idx):
        return self.trajectories_r1[idx], self.trajectories_r2[idx]