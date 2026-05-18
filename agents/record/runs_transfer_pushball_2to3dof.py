"""
runs_transfer_pushball_2to3dof.py
Collecte 20 épisodes réussis et 20 échoués — politique pushball 2-DOF transférée → env 3-DOF.
Sauvegarde dans :
  runs/transfer_pushball_2to3dof/success/ep_XXX.npz
  runs/transfer_pushball_2to3dof/fail/ep_XXX.npz
"""

import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_3dof import PushBallEnv_3dof
from agents.test.transfer_pushball_2to3dof import TransferPolicy

# ── Configuration ──────────────────────────────────────────────────────────────
RUN_ID_2DOF      = 1
SAVE_TRANSFER_ID = 1

POLICY_2DOF_PATH  = f"./models/ppo_pushball_2dof_{RUN_ID_2DOF}/best_model.zip"
VECNORM_2DOF_PATH = f"./models/ppo_pushball_2dof_{RUN_ID_2DOF}/vec_normalize.pkl"
STATE_MAPPER_PATH  = "./data/DIRECT/state_mapper_3to2dof.pt"   # 8 → 6
ACTION_MAPPER_PATH = "./data/DIRECT/action_mapper_2to3dof.pt"  # (6+2)→3

N_TARGET  = 20
MAX_STEPS = 150

SAVE_SUCCESS = "runs/transfer_pushball_2to3dof/success"
SAVE_FAIL    = "runs/transfer_pushball_2to3dof/fail"
os.makedirs(SAVE_SUCCESS, exist_ok=True)
os.makedirs(SAVE_FAIL,    exist_ok=True)

# ── Chargement politique de transfert ─────────────────────────────────────────
policy = TransferPolicy(
    policy_2dof_path  = POLICY_2DOF_PATH,
    vecnorm_2dof_path = VECNORM_2DOF_PATH,
    state_mapper_path  = STATE_MAPPER_PATH,
    action_mapper_path = ACTION_MAPPER_PATH,
    device = "cpu",
)

def make_env():
    return Monitor(PushBallEnv_3dof(render_mode=None, max_steps=MAX_STEPS))

env   = DummyVecEnv([make_env])
inner = env.envs[0].unwrapped

# ── Collecte ───────────────────────────────────────────────────────────────────
n_success = 0
n_fail    = 0
ep        = 0

print(f"Collecte de {N_TARGET} succès et {N_TARGET} échecs — transfer pushball 2→3 DOF\n")

while n_success < N_TARGET or n_fail < N_TARGET:
    obs = env.reset()
    target    = inner.target.copy()
    ball_init = inner.ball.copy()

    thetas         = []
    balls          = []
    rewards        = []
    dists_ball_tgt = []

    done = False
    info_last = {}

    while not done:
        action_3dof = policy.predict(obs[0])
        obs, reward, dones, infos = env.step(action_3dof)

        thetas.append([inner.theta1, inner.theta2, inner.theta3])
        balls.append(inner.ball.copy())
        rewards.append(float(reward[0]))
        dists_ball_tgt.append(float(infos[0].get("dist_ball_target", 0.0)))

        done      = dones[0]
        info_last = infos[0]

    ep += 1
    success = info_last.get("target_reached", False)

    data = dict(
        target         = target,
        ball_init      = ball_init,
        thetas         = np.array(thetas,         dtype=np.float32),
        balls          = np.array(balls,           dtype=np.float32),
        rewards        = np.array(rewards,         dtype=np.float32),
        dists_ball_tgt = np.array(dists_ball_tgt, dtype=np.float32),
        success        = success,
        total_reward   = float(np.sum(rewards)),
        l1             = inner.l1,
        l2             = inner.l2,
        l3             = inner.l3,
        epsilon        = inner.epsilon,
        eff_radius     = inner.eff_radius,
        ball_radius    = inner.ball_radius,
        max_reach      = inner.max_reach,
    )

    if success and n_success < N_TARGET:
        np.savez(os.path.join(SAVE_SUCCESS, f"ep_{n_success:03d}.npz"), **data)
        n_success += 1
        print(f"  [ep {ep:4d}]  ✓  succès #{n_success:2d}/{N_TARGET}"
              f"  |  steps={len(thetas):3d}  reward={data['total_reward']:+.1f}")
    elif not success and n_fail < N_TARGET:
        np.savez(os.path.join(SAVE_FAIL, f"ep_{n_fail:03d}.npz"), **data)
        n_fail += 1
        print(f"  [ep {ep:4d}]  ✗  échec  #{n_fail:2d}/{N_TARGET}"
              f"  |  steps={len(thetas):3d}  dist_finale={dists_ball_tgt[-1]:.4f} m")

print(f"\n✔  Terminé : {n_success} succès, {n_fail} échecs en {ep} épisodes.")
env.close()
