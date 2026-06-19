"""
runs_reach_3dof.py
Collecte 20 épisodes réussis et 20 épisodes échoués pour le reaching 3-DOF.
Sauvegarde les trajectoires complètes dans :
  runs/reach_3dof/success/ep_XXX.npz
  runs/reach_3dof/fail/ep_XXX.npz
"""

import os
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.env_reaching_3dof import ReachingEnv_3dof

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "data" / "models").exists())
MODEL_ROOT = ROOT / "data" / "models"

# ── Configuration ──────────────────────────────────────────────────────────────
RUN_ID       = 1
MODEL_PATH   = MODEL_ROOT / f"ppo_reach_3dof_{RUN_ID}" / "best_model.zip"
VECNORM_PATH = MODEL_ROOT / f"ppo_reach_3dof_{RUN_ID}" / "vec_normalize.pkl"
MAX_STEPS    = 100
N_TARGET     = 20

SAVE_SUCCESS = "runs/reach_3dof/success"
SAVE_FAIL    = "runs/reach_3dof/fail"
os.makedirs(SAVE_SUCCESS, exist_ok=True)
os.makedirs(SAVE_FAIL,    exist_ok=True)

# ── Chargement modèle ──────────────────────────────────────────────────────────
model = PPO.load(MODEL_PATH, custom_objects={"learning_rate": 0.0003, "lr_schedule": lambda _: 0.0003, "clip_range": lambda _: 0.2})

def make_env():
    return Monitor(ReachingEnv_3dof(render_mode=None, max_steps=MAX_STEPS))

env = DummyVecEnv([make_env])
env = VecNormalize.load(VECNORM_PATH, env)
env.training    = False
env.norm_reward = False

inner = env.envs[0].unwrapped

# ── Collecte ───────────────────────────────────────────────────────────────────
n_success = 0
n_fail    = 0
ep        = 0

print(f"Collecte de {N_TARGET} succès et {N_TARGET} échecs — reaching 3-DOF\n")

while n_success < N_TARGET or n_fail < N_TARGET:
    obs = env.reset()
    target = inner.target.copy()

    thetas  = []   # (T, 3)
    rewards = []
    dists   = []

    done = False
    info_last = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)

        thetas.append([infos[0]["theta1"], infos[0]["theta2"], infos[0]["theta3"]])
        rewards.append(float(reward[0]))
        dists.append(float(infos[0].get("dist", 0.0)))

        done      = dones[0]
        info_last = infos[0]

    ep += 1
    success = info_last.get("target_reached", False)

    data = dict(
        target       = target,
        thetas       = np.array(thetas,  dtype=np.float32),   # (T, 3)
        rewards      = np.array(rewards, dtype=np.float32),
        dists        = np.array(dists,   dtype=np.float32),
        success      = success,
        total_reward = float(np.sum(rewards)),
        l1           = inner.l1,
        l2           = inner.l2,
        l3           = inner.l3,
        epsilon      = inner.epsilon,
    )

    if success and n_success < N_TARGET:
        n_success += 1
        path = os.path.join(SAVE_SUCCESS, f"ep_{n_success:03d}.npz")
        np.savez(path, **data)
        print(f"  [ep {ep:4d}]  ✓  succès #{n_success:2d}/{N_TARGET}"
              f"  |  steps={len(thetas):3d}  reward={data['total_reward']:+.1f}")

    elif not success and n_fail < N_TARGET:
        n_fail += 1
        path = os.path.join(SAVE_FAIL, f"ep_{n_fail:03d}.npz")
        np.savez(path, **data)
        print(f"  [ep {ep:4d}]  ✗  échec  #{n_fail:2d}/{N_TARGET}"
              f"  |  steps={len(thetas):3d}  dist_finale={dists[-1]:.4f} m")

print(f"\n✔  Terminé : {n_success} succès, {n_fail} échecs collectés en {ep} épisodes.")
env.close()
