import numpy as np
from scipy.interpolate import interp1d


def spatial_sampling(trajectory: np.ndarray, num_samples: int = 50) -> np.ndarray:
    """
    Ré-échantillonne une trajectoire (T, D) en num_samples points espacés
    régulièrement en distance curviligne (arc-length).

    Fix : gère le cas dégénéré où tous les points sont identiques
    (total_dist ≈ 0) en renvoyant un sous-échantillonnage uniforme en temps.
    """
    diffs = np.diff(trajectory, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cum_dist[-1]

    # --- Cas dégénéré : trajectoire plate ---
    if total_dist < 1e-9:
        idx = np.round(np.linspace(0, len(trajectory) - 1, num_samples)).astype(int)
        return trajectory[idx].astype(np.float32)

    new_cum_dist = np.linspace(0, total_dist, num_samples)
    new_traj = np.zeros((num_samples, trajectory.shape[1]), dtype=np.float32)
    for d in range(trajectory.shape[1]):
        interp = interp1d(cum_dist, trajectory[:, d], kind='linear',
                          fill_value='extrapolate')
        new_traj[:, d] = interp(new_cum_dist)
    return new_traj


def spatial_sampling_aligned(states: np.ndarray, actions: np.ndarray,
                              num_samples: int = 50):
    """
    Ré-échantillonne CONJOINTEMENT states (T, Ds) et actions (T, Da) en
    num_samples points, en utilisant la paramétrisation arc-length calculée
    sur les STATES uniquement.

    Cela garantit que le point rééchantillonné states[i] correspond bien au
    point rééchantillonné actions[i], préservant l'alignement temporel
    (état_t, action_t) indispensable pour l'entraînement du mapper.

    Précondition : len(states) == len(actions)  (après trim dans record_one_segment)

    Retourne : (states_ss, actions_ss)  toutes deux de shape (num_samples, .)
    """
    assert len(states) == len(actions), (
        f"states ({len(states)}) et actions ({len(actions)}) doivent avoir "
        "le même nombre de lignes après alignement."
    )

    # Arc-length calculé depuis l'espace des états
    diffs = np.diff(states, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cum_dist[-1]

    # --- Cas dégénéré ---
    if total_dist < 1e-9:
        idx = np.round(np.linspace(0, len(states) - 1, num_samples)).astype(int)
        return states[idx].astype(np.float32), actions[idx].astype(np.float32)

    target_dists = np.linspace(0, total_dist, num_samples)

    states_out  = np.zeros((num_samples, states.shape[1]),  dtype=np.float32)
    actions_out = np.zeros((num_samples, actions.shape[1]), dtype=np.float32)

    for d in range(states.shape[1]):
        interp = interp1d(cum_dist, states[:, d], kind='linear',
                          fill_value='extrapolate')
        states_out[:, d] = interp(target_dists)

    for d in range(actions.shape[1]):
        interp = interp1d(cum_dist, actions[:, d], kind='linear',
                          fill_value='extrapolate')
        actions_out[:, d] = interp(target_dists)

    return states_out, actions_out
