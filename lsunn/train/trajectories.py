# lsunn/train/trajectories.py
"""
Phase 1 : Collecte de trajectoires aléatoires appairées pour 2DoF et 3DoF.
Sauvegarde le dataset dans DATA_DIR/trajectories.pkl.
"""

import sys
import numpy as np
import pickle
import gc
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof

# ── Dimensions ────────────────────────────────────────────────────────────────
ARM_OBS_2DOF   = 6
ARM_OBS_3DOF   = 8
ACTION_DIM_2DOF = 2
ACTION_DIM_3DOF = 3

# ── Chemins ───────────────────────────────────────────────────────────────────
DATA_DIR  = Path("./data/LSUNN/joint_model")
TRAJ_PATH = DATA_DIR / "trajectories.pkl"


# ── Collecte ──────────────────────────────────────────────────────────────────

def collect_random_paired_trajectories(n_pairs: int = 10_000, seq_len: int = 30) -> dict:
    """
    Collecte ``n_pairs`` paires de trajectoires synchronisées (même config
    initiale) avec des actions aléatoires dans les deux environnements.

    Returns
    -------
    dict avec les clés :
        states_2dof, states_3dof,
        arm_states_2dof, arm_states_3dof,
        actions_2dof,  actions_3dof
    """
    print("\n" + "=" * 60)
    print("Phase 1: Collecting random paired trajectories")
    print("=" * 60)

    env2 = PushBallEnv_2dof(render_mode=None, max_steps=1_000_000)
    env3 = PushBallEnv_3dof(render_mode=None, max_steps=1_000_000)
    rng  = np.random.RandomState(42)

    max_reach = env2.max_reach
    s2_full_dim = env2.observation_space.shape[0]
    s3_full_dim = env3.observation_space.shape[0]

    states2_full, states3_full = [], []
    arm_states2,  arm_states3  = [], []
    acts2,        acts3        = [], []

    for _ in tqdm(range(n_pairs), desc="Random pairs"):
        # ── Configuration initiale commune ────────────────────────────────
        r_target = rng.uniform(0.4, 0.75, 2) * max_reach
        angle    = rng.uniform(-np.pi, np.pi)
        target   = np.array([r_target[0] * np.cos(angle),
                              r_target[0] * np.sin(angle)])

        # Balle à distance >= 0.3 du target
        for _ in range(100):
            r_ball   = rng.uniform(0.3, 0.75, 2) * max_reach
            angle_b  = rng.uniform(-np.pi, np.pi)
            ball     = np.array([r_ball[0] * np.cos(angle_b),
                                  r_ball[0] * np.sin(angle_b)])
            if np.linalg.norm(ball - target) >= 0.3:
                break

        env2.reset(); env3.reset()
        env2.target, env2.ball = target.copy(), ball.copy()
        env3.target, env3.ball = target.copy(), ball.copy()

        # ── Roulement de trajectoire ───────────────────────────────────────
        traj2_full, traj3_full = [], []
        traj2_arm,  traj3_arm  = [], []
        act_traj2,  act_traj3  = [], []

        for _ in range(seq_len):
            a2 = rng.uniform(-1, 1, ACTION_DIM_2DOF).astype(np.float32)
            a3 = rng.uniform(-1, 1, ACTION_DIM_3DOF).astype(np.float32)

            obs2, _, _, _, _ = env2.step(a2)
            obs3, _, _, _, _ = env3.step(a3)

            traj2_full.append(obs2)
            traj3_full.append(obs3)
            traj2_arm.append(obs2[:ARM_OBS_2DOF])
            traj3_arm.append(obs3[:ARM_OBS_3DOF])
            act_traj2.append(a2)
            act_traj3.append(a3)

        states2_full.append(np.stack(traj2_full))
        states3_full.append(np.stack(traj3_full))
        arm_states2.append(np.stack(traj2_arm))
        arm_states3.append(np.stack(traj3_arm))
        acts2.append(np.stack(act_traj2))
        acts3.append(np.stack(act_traj3))

    env2.close()
    env3.close()

    result = {
        "states_2dof":     np.stack(states2_full).reshape(-1, s2_full_dim),
        "states_3dof":     np.stack(states3_full).reshape(-1, s3_full_dim),
        "arm_states_2dof": np.stack(arm_states2).reshape(-1, ARM_OBS_2DOF),
        "arm_states_3dof": np.stack(arm_states3).reshape(-1, ARM_OBS_3DOF),
        "actions_2dof":    np.stack(acts2).reshape(-1, ACTION_DIM_2DOF),
        "actions_3dof":    np.stack(acts3).reshape(-1, ACTION_DIM_3DOF),
    }

    del states2_full, states3_full, arm_states2, arm_states3, acts2, acts3
    gc.collect()

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = collect_random_paired_trajectories(n_pairs=10_000, seq_len=30)

    with open(TRAJ_PATH, "wb") as f:
        pickle.dump(data, f)

    print(f"\n  Trajectories saved → {TRAJ_PATH}")
    for k, v in data.items():
        print(f"    {k}: {v.shape}")


if __name__ == "__main__":
    main()
