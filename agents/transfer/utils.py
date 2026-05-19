import numpy as np
from scipy.interpolate import interp1d

def spatial_sampling(trajectory: np.ndarray, num_samples: int = 50) -> np.ndarray:
    """
    Ré‑échantillonne une trajectoire (T, D) en N points espacés régulièrement
    en distance curviligne (arc‑length).
    """
    diffs = np.diff(trajectory, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cum_dist[-1]

    new_cum_dist = np.linspace(0, total_dist, num_samples)

    new_traj = np.zeros((num_samples, trajectory.shape[1]))
    for d in range(trajectory.shape[1]):
        interp = interp1d(cum_dist, trajectory[:, d], kind='linear',
                          fill_value='extrapolate')
        new_traj[:, d] = interp(new_cum_dist)
    return new_traj
