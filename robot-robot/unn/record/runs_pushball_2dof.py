"""
runs_pushball_2dof.py  (UNN)
Collecte N_TARGET épisodes réussis et N_TARGET échoués — politique UNN pushball 2-DOF.
Sauvegarde dans :
  runs/unn_pushball_2dof/success/ep_XXX.npz
  runs/unn_pushball_2dof/fail/ep_XXX.npz
"""

import sys
import os
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_2dof import PushBallEnv_2dof
from unn.bases_unn import CartesianStateEncoder
from unn.unn_policy import UNNPolicy

# ── Configuration ──────────────────────────────────────────────────────────────
POLICY_DIR  = "./data/UNN/unn_pushball_2dof"
POLICY_NAME = "final"
MAX_STEPS   = 150
N_TARGET    = 20
DEVICE      = "cpu"

SAVE_SUCCESS = "runs/unn_pushball_2dof/success"
SAVE_FAIL    = "runs/unn_pushball_2dof/fail"
os.makedirs(SAVE_SUCCESS, exist_ok=True)
os.makedirs(SAVE_FAIL,    exist_ok=True)

# ── Chargement politique UNN ───────────────────────────────────────────────────
encoder = CartesianStateEncoder(max_reach=3.0)
policy  = UNNPolicy.load(POLICY_DIR, name=POLICY_NAME, encoder=encoder, device=DEVICE)
print(f"UNN 2-DoF policy loaded from '{POLICY_DIR}/{POLICY_NAME}_*'")

# ── Environnement ──────────────────────────────────────────────────────────────
def make_env():
    return Monitor(PushBallEnv_2dof(render_mode=None, max_steps=MAX_STEPS))

env   = DummyVecEnv([make_env])
inner = env.envs[0].unwrapped

# ── Collecte ───────────────────────────────────────────────────────────────────
n_success = 0
n_fail    = 0
ep        = 0

print(f"Collecte de {N_TARGET} succès et {N_TARGET} échecs — UNN push-ball 2-DOF\n")

while n_success < N_TARGET or n_fail < N_TARGET:
    obs       = env.reset()[0]         # (obs_dim,)
    target    = inner.target.copy()
    ball_init = inner.ball.copy()

    thetas         = []
    balls          = []
    rewards        = []
    dists_ball_tgt = []

    done      = False
    info_last = {}

    while not done:
        action = policy.predict(obs, deterministic=True)              # (2,)
        obs, reward, dones, infos = env.step(action.reshape(1, -1))
        obs = obs[0]                                                  # unwrap batch dim

        thetas.append([inner.theta1, inner.theta2])
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
        thetas         = np.array(thetas,         dtype=np.float32),   # (T, 2)
        balls          = np.array(balls,           dtype=np.float32),   # (T, 2)
        rewards        = np.array(rewards,         dtype=np.float32),
        dists_ball_tgt = np.array(dists_ball_tgt, dtype=np.float32),
        success        = success,
        total_reward   = float(np.sum(rewards)),
        l1             = inner.l1,
        l2             = inner.l2,
        epsilon        = inner.epsilon,
        eff_radius     = inner.eff_radius,
        ball_radius    = inner.ball_radius,
        max_reach      = inner.max_reach,
    )

    if success and n_success < N_TARGET:
        n_success += 1
        np.savez(os.path.join(SAVE_SUCCESS, f"ep_{n_success:03d}.npz"), **data)
        print(f"  [ep {ep:4d}]  ✓  succès #{n_success:2d}/{N_TARGET}"
              f"  |  steps={len(thetas):3d}  reward={data['total_reward']:+.1f}")
    elif not success and n_fail < N_TARGET:
        n_fail += 1
        np.savez(os.path.join(SAVE_FAIL, f"ep_{n_fail:03d}.npz"), **data)
        print(f"  [ep {ep:4d}]  ✗  échec  #{n_fail:2d}/{N_TARGET}"
              f"  |  steps={len(thetas):3d}  dist_finale={dists_ball_tgt[-1]:.4f} m")

print(f"\n✔  Terminé : {n_success} succès, {n_fail} échecs en {ep} épisodes.")
env.close()
