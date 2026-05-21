"""
Collect paired trajectories for UNN: raw states, joint actions,
and the resulting end‑effector displacement (to train action mappers).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof

# ---------- Helper : forward kinematics for end‑effector position ----------
def ee_position_2dof(theta1, theta2, l1=1.5, l2=1.5):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return np.array([x, y])

def ee_position_3dof(theta1, theta2, theta3, l1=1.0, l2=1.0, l3=1.0):
    x = (l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) +
         l3 * np.cos(theta1 + theta2 + theta3))
    y = (l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) +
         l3 * np.sin(theta1 + theta2 + theta3))
    return np.array([x, y])

# ---------- Load pre‑trained PPO policies (source robots) ----------
def load_policy(model_path, vec_path, env_class):
    env = env_class(render_mode=None)
    model = PPO.load(model_path, device="cpu")
    venv = DummyVecEnv([lambda: Monitor(env_class(render_mode=None))])
    vec_norm = VecNormalize.load(vec_path, venv=venv)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return model, vec_norm

# ---------- Main ----------
def main():
    RUN_ID_2 = 1
    RUN_ID_3 = 1
    MODEL_2 = f"./models/ppo_pushball_2dof_{RUN_ID_2}/best_model.zip"
    VEC_2   = f"./models/ppo_pushball_2dof_{RUN_ID_2}/vec_normalize.pkl"
    MODEL_3 = f"./models/ppo_pushball_3dof_{RUN_ID_3}/best_model.zip"
    VEC_3   = f"./models/ppo_pushball_3dof_{RUN_ID_3}/vec_normalize.pkl"

    policy2, _ = load_policy(MODEL_2, VEC_2, PushBallEnv_2dof)
    policy3, _ = load_policy(MODEL_3, VEC_3, PushBallEnv_3dof)

    N_PAIRS = 2000           # nombre de segments
    STEPS_PER_PAIR = 200     # longueur de chaque segment
    SEED = 42

    data_2dof = []   # list of (obs, action, joint_angles, ee_disp)
    data_3dof = []

    rng = np.random.RandomState(SEED)
    max_reach = 3.0

    for pair_idx in tqdm(range(N_PAIRS), desc="Collecting pairs"):
        env2 = PushBallEnv_2dof(render_mode=None, max_steps=1_000_000)
        env3 = PushBallEnv_3dof(render_mode=None, max_steps=1_000_000)
        env2.reset(seed=SEED + pair_idx)
        env3.reset(seed=SEED + pair_idx)

        # Force same target/ball for both environments
        target = env2.target.copy()
        ball   = env2.ball.copy()
        env3.target = target.copy()
        env3.ball   = ball.copy()

        obs2 = env2._get_obs()
        obs3 = env3._get_obs()

        # Get initial joint angles (radians) from environment
        theta1_2, theta2_2 = env2.theta1, env2.theta2
        theta1_3, theta2_3, theta3_3 = env3.theta1, env3.theta2, env3.theta3
        ee_prev2 = ee_position_2dof(theta1_2, theta2_2)
        ee_prev3 = ee_position_3dof(theta1_3, theta2_3, theta3_3)

        for step in range(STEPS_PER_PAIR):
            # --- 2DoF ---
            act2, _ = policy2.predict(vec_norm2.normalize_obs(obs2.reshape(1,-1)), deterministic=True)
            act2 = act2[0]
            new_theta1_2 = np.clip(theta1_2 + act2[0]*0.1, -np.pi, np.pi)
            new_theta2_2 = np.clip(theta2_2 + act2[1]*0.1, -np.pi, np.pi)
            ee_new2 = ee_position_2dof(new_theta1_2, new_theta2_2)
            ee_disp2 = ee_new2 - ee_prev2
            # Store: (obs, action, joint_angles_norm, desired_disp_norm)
            joint_norm2 = np.array([theta1_2/np.pi, theta2_2/np.pi], dtype=np.float32)
            disp_norm2 = np.clip(ee_disp2 / 0.2, -1.0, 1.0)   # max disp ~0.2 rad*1.5*2? keep safe
            data_2dof.append((obs2.copy(), act2, joint_norm2, disp_norm2))

            # Step environment
            obs2, _, _, _, _ = env2.step(act2)
            theta1_2, theta2_2 = new_theta1_2, new_theta2_2
            ee_prev2 = ee_new2

            # --- 3DoF ---
            act3, _ = policy3.predict(vec_norm3.normalize_obs(obs3.reshape(1,-1)), deterministic=True)
            act3 = act3[0]
            new_theta1_3 = np.clip(theta1_3 + act3[0]*0.1, -np.pi, np.pi)
            new_theta2_3 = np.clip(theta2_3 + act3[1]*0.1, -np.pi, np.pi)
            new_theta3_3 = np.clip(theta3_3 + act3[2]*0.1, -np.pi, np.pi)
            ee_new3 = ee_position_3dof(new_theta1_3, new_theta2_3, new_theta3_3)
            ee_disp3 = ee_new3 - ee_prev3
            joint_norm3 = np.array([theta1_3/np.pi, theta2_3/np.pi, theta3_3/np.pi], dtype=np.float32)
            disp_norm3 = np.clip(ee_disp3 / 0.2, -1.0, 1.0)
            data_3dof.append((obs3.copy(), act3, joint_norm3, disp_norm3))

            obs3, _, _, _, _ = env3.step(act3)
            theta1_3, theta2_3, theta3_3 = new_theta1_3, new_theta2_3, new_theta3_3
            ee_prev3 = ee_new3

        env2.close()
        env3.close()

    # Convert to numpy arrays
    def extract(arr):
        obs = np.stack([x[0] for x in arr])
        act = np.stack([x[1] for x in arr])
        joint = np.stack([x[2] for x in arr])
        disp = np.stack([x[3] for x in arr])
        return obs, act, joint, disp

    obs2, act2, joint2, disp2 = extract(data_2dof)
    obs3, act3, joint3, disp3 = extract(data_3dof)

    trajectories = {
        '2dof': {'obs': obs2, 'action': act2, 'joint_norm': joint2, 'ee_disp_norm': disp2},
        '3dof': {'obs': obs3, 'action': act3, 'joint_norm': joint3, 'ee_disp_norm': disp3},
        'metadata': {'n_samples': len(obs2), 'steps_per_pair': STEPS_PER_PAIR}
    }

    save_path = Path("./data/UNN/trajectories_unn.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(obs2)} samples to {save_path}")

if __name__ == "__main__":
    main()
