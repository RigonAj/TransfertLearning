import os
import pickle
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_continuous_reaching_2dof import Arm2DoFPersistentEnv
from envs.env_continuous_reaching_3dof import Arm3DoFPersistentEnv
from agents.transfer.utils import spatial_sampling   # <-- notre fonction SS

import torch
torch.set_num_threads(4)

# =========================================================
# Helpers (inchangés, mais je les laisse pour être complet)
# =========================================================

def load_policy_and_vecnorm(model_path, vec_path, env_factory, device="cpu"):
    model = PPO.load(model_path, device=device, custom_objects={
        "learning_rate": 0.0003,
        "lr_schedule": lambda _: 0.0003,
        "clip_range": lambda _: 0.2
    })
    venv = DummyVecEnv([env_factory])
    vec_norm = VecNormalize.load(vec_path, venv=venv)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return model, vec_norm, venv

def predict_action(model, vec_norm, raw_obs):
    obs_norm = vec_norm.normalize_obs(raw_obs.reshape(1, -1))
    action, _ = model.predict(obs_norm, deterministic=True)
    return action[0]

def override_target(env, target):
    if hasattr(env, 'set_target'):
        env.set_target(target)
    else:
        env.target = target.copy()
        if hasattr(env, 'theta3'):
            ee = env.forward_kinematics(env.theta1, env.theta2, env.theta3)
        else:
            ee = env.forward_kinematics(env.theta1, env.theta2)
        env.prev_dist = float(np.linalg.norm(ee - target))

def get_ee(env):
    if hasattr(env, 'theta3'):
        return env.forward_kinematics(env.theta1, env.theta2, env.theta3)
    else:
        return env.forward_kinematics(env.theta1, env.theta2)

def generate_random_target(rng, max_reach, scale=0.98):
    r = rng.uniform(0.05, max_reach * scale)
    angle = rng.uniform(-np.pi, np.pi)
    return np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)

def generate_target_sequence(rng, length, max_reach, scale=0.98):
    return [generate_random_target(rng, max_reach, scale) for _ in range(length)]

def both_close_to_target(env2, env3, target, tol=0.2):
    ee2 = get_ee(env2)
    ee3 = get_ee(env3)
    return (np.linalg.norm(ee2 - target) < tol and
            np.linalg.norm(ee3 - target) < tol)

def wait_until_both_close(env2, env3, target, policy2, vecnorm2, policy3, vecnorm3,
                          max_steps=500, tol=0.2):
    override_target(env2, target)
    override_target(env3, target)
    steps = 0
    while steps < max_steps and not both_close_to_target(env2, env3, target, tol):
        act2 = predict_action(policy2, vecnorm2, env2._get_obs())
        act3 = predict_action(policy3, vecnorm3, env3._get_obs())
        env2.step(act2)
        env3.step(act3)
        steps += 1
    return both_close_to_target(env2, env3, target, tol), steps

def record_one_segment(env2, env3, start_target, end_target,
                       policy2, vecnorm2, policy3, vecnorm3,
                       max_steps=300, tol=0.2):
    """
    Retourne (states2, states3, ee2, ee3, actions2, actions3, success, info_dict)
    """
    if not both_close_to_target(env2, env3, start_target, tol):
        ok, _ = wait_until_both_close(env2, env3, start_target, policy2, vecnorm2, policy3, vecnorm3,
                                      max_steps=100, tol=tol)
        if not ok:
            return None, None, None, None, None, None, False, {"reason": "start_not_close"}

    override_target(env2, end_target)
    override_target(env3, end_target)

    arm_size2 = env2.arm_obs_size
    arm_size3 = env3.arm_obs_size

    states2 = [env2._get_obs()[:arm_size2].copy()]
    states3 = [env3._get_obs()[:arm_size3].copy()]
    actions2 = []
    actions3 = []
    ee2 = [get_ee(env2).copy()]
    ee3 = [get_ee(env3).copy()]

    reached2 = False
    reached3 = False
    steps = 0

    while steps < max_steps and not (reached2 and reached3):
        act2 = predict_action(policy2, vecnorm2, env2._get_obs()) if not reached2 else np.zeros(2)
        act3 = predict_action(policy3, vecnorm3, env3._get_obs()) if not reached3 else np.zeros(3)

        if not reached2:
            actions2.append(act2.copy())
        if not reached3:
            actions3.append(act3.copy())

        env2.step(act2)
        env3.step(act3)

        obs2 = env2._get_obs()
        obs3 = env3._get_obs()
        if not reached2:
            states2.append(obs2[:arm_size2].copy())
            ee2.append(get_ee(env2).copy())
        if not reached3:
            states3.append(obs3[:arm_size3].copy())
            ee3.append(get_ee(env3).copy())

        if not reached2 and np.linalg.norm(ee2[-1] - end_target) < tol:
            reached2 = True
        if not reached3 and np.linalg.norm(ee3[-1] - end_target) < tol:
            reached3 = True

        steps += 1

    success = reached2 and reached3
    info = {
        "steps": steps,
        "dist2_final": np.linalg.norm(ee2[-1] - end_target) if len(ee2) > 0 else 999,
        "dist3_final": np.linalg.norm(ee3[-1] - end_target) if len(ee3) > 0 else 999,
        "reached2": reached2,
        "reached3": reached3,
    }
    states2 = np.array(states2, dtype=np.float32)
    states3 = np.array(states3, dtype=np.float32)
    actions2 = np.array(actions2, dtype=np.float32) if actions2 else np.empty((0,2))
    actions3 = np.array(actions3, dtype=np.float32) if actions3 else np.empty((0,3))
    ee2 = np.array(ee2, dtype=np.float32)
    ee3 = np.array(ee3, dtype=np.float32)

    return states2, states3, ee2, ee3, actions2, actions3, success, info

# =========================================================
# Génération principale – AVEC SPATIAL SAMPLING
# =========================================================
def main():
    NUM_SAMPLES = 50                     # <-- longueur fixe après SS
    N_SEGMENTS_TARGET = 10_000
    TARGETS_PER_EPISODE = 30
    MAX_STEPS_PER_SEGMENT = 60
    TOLERANCE = 0.2
    WAIT_MAX_STEPS = 500
    TARGET_SCALE = 0.98

    data_dir = Path("./data/DIRECT")
    data_dir.mkdir(parents=True, exist_ok=True)
    traj_path = data_dir / "trajectories_aligned_ss.pkl"   # nouveau nom

    run_id_2dof = 1
    run_id_3dof = 1
    MODEL_2DOF = f"./models/ppo_reach_2dof_{run_id_2dof}/best_model.zip"
    VECNORM_2DOF = f"./models/ppo_reach_2dof_{run_id_2dof}/vec_normalize.pkl"
    MODEL_3DOF = f"./models/ppo_reach_3dof_{run_id_3dof}/best_model.zip"
    VECNORM_3DOF = f"./models/ppo_reach_3dof_{run_id_3dof}/vec_normalize.pkl"

    def make_2dof():
        return Monitor(Arm2DoFPersistentEnv(render_mode=None))
    def make_3dof():
        return Monitor(Arm3DoFPersistentEnv(render_mode=None))

    device = "cpu"
    policy2, vecnorm2, _ = load_policy_and_vecnorm(MODEL_2DOF, VECNORM_2DOF, make_2dof, device)
    policy3, vecnorm3, _ = load_policy_and_vecnorm(MODEL_3DOF, VECNORM_3DOF, make_3dof, device)

    max_reach = min(Arm2DoFPersistentEnv().max_reach, Arm3DoFPersistentEnv().max_reach)
    rng = np.random.RandomState(42)

    # On va stocker des listes de segments (chacun déjà ré‑échantillonné)
    all_segments_s2 = []   # chaque élément: array (NUM_SAMPLES, dim2)
    all_segments_s3 = []   # (NUM_SAMPLES, dim3)
    all_segments_a2 = []   # (NUM_SAMPLES, 2)
    all_segments_a3 = []   # (NUM_SAMPLES, 3)

    valid_segments = 0
    start_time = time.time()
    episode = 0

    pbar = tqdm(total=N_SEGMENTS_TARGET, desc="Valid segments (SS)", unit="seg")

    while valid_segments < N_SEGMENTS_TARGET:
        episode += 1
        targets = generate_target_sequence(rng, TARGETS_PER_EPISODE, max_reach, scale=TARGET_SCALE)

        env2 = Arm2DoFPersistentEnv(render_mode=None)
        env3 = Arm3DoFPersistentEnv(render_mode=None)
        env2.max_steps = 1_000_000
        env3.max_steps = 1_000_000
        env2.reset(seed=episode)
        env3.reset(seed=episode)

        env2.theta1 = 0.0; env2.theta2 = 0.0
        env3.theta1 = 0.0; env3.theta2 = 0.0; env3.theta3 = 0.0
        env2.dtheta1 = 0.0; env2.dtheta2 = 0.0
        env3.dtheta1 = 0.0; env3.dtheta2 = 0.0; env3.dtheta3 = 0.0
        override_target(env2, targets[0])
        override_target(env3, targets[0])

        ok, steps_taken = wait_until_both_close(env2, env3, targets[0],
                                                policy2, vecnorm2, policy3, vecnorm3,
                                                max_steps=WAIT_MAX_STEPS, tol=TOLERANCE)
        if not ok:
            print(f"Épisode {episode} échoué (première cible)")
            env2.close()
            env3.close()
            continue

        for i in range(len(targets)-1):
            start_tgt = targets[i]
            end_tgt = targets[i+1]

            s2, s3, ee2, ee3, a2, a3, success, info = record_one_segment(
                env2, env3, start_tgt, end_tgt,
                policy2, vecnorm2, policy3, vecnorm3,
                max_steps=MAX_STEPS_PER_SEGMENT, tol=TOLERANCE
            )
            if not success:
                continue

            # ---------- Application du Spatial Sampling ----------
            s2_ss = spatial_sampling(s2, NUM_SAMPLES)
            s3_ss = spatial_sampling(s3, NUM_SAMPLES)
            a2_ss = spatial_sampling(a2, NUM_SAMPLES)
            a3_ss = spatial_sampling(a3, NUM_SAMPLES)

            all_segments_s2.append(s2_ss)
            all_segments_s3.append(s3_ss)
            all_segments_a2.append(a2_ss)
            all_segments_a3.append(a3_ss)

            valid_segments += 1
            pbar.update(1)

            elapsed = time.time() - start_time
            rate = valid_segments / elapsed if elapsed > 0 else 0
            pbar.set_postfix({"rate": f"{rate:.2f} seg/s"})

            if valid_segments >= N_SEGMENTS_TARGET:
                break

        env2.close()
        env3.close()

    pbar.close()

    # Conversion en tableaux numpy : shape (n_segments, NUM_SAMPLES, dim)
    segments_s2 = np.stack(all_segments_s2, axis=0)
    segments_s3 = np.stack(all_segments_s3, axis=0)
    segments_a2 = np.stack(all_segments_a2, axis=0)
    segments_a3 = np.stack(all_segments_a3, axis=0)

    print(f"\nGenerated {len(segments_s2)} segments of length {NUM_SAMPLES} each.")
    print(f"Total samples (pas de temps) : {segments_s2.shape[0] * segments_s2.shape[1]}")

    trajectories = {
        'segments_2dof': segments_s2,   # (N, L, 6)
        'segments_3dof': segments_s3,   # (N, L, 8)
        'segments_actions_2dof': segments_a2,  # (N, L, 2)
        'segments_actions_3dof': segments_a3,  # (N, L, 3)
        'metadata': {
            'n_segments': valid_segments,
            'seq_len': NUM_SAMPLES,
            'source': 'spatial_sampling',
        }
    }
    with open(traj_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"\nSaved -> {traj_path}")

if __name__ == "__main__":
    main()
