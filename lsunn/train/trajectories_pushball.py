"""
Collect paired, time-aligned trajectories for PushBall 2-DoF and 3-DoF.

This is similar to agents/transfer/trajectories.py but adapted for
pushball task and LS-UNN requirements (full state = arm_obs + task_obs).

Usage:
    python -m lsunn.trajectories_pushball
"""

import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof

import torch
torch.set_num_threads(4)


def load_policy_and_vecnorm(model_path, vec_path, env_factory, device="cpu"):
    model = PPO.load(model_path, device=device)
    venv = DummyVecEnv([env_factory])
    vec_norm = VecNormalize.load(vec_path, venv=venv)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return model, vec_norm, venv


def predict_action(model, vec_norm, raw_obs):
    obs_norm = vec_norm.normalize_obs(raw_obs.reshape(1, -1))
    action, _ = model.predict(obs_norm, deterministic=True)
    return action[0]


def override_target_and_ball(env_raw, target, ball):
    """Set target and ball position for persistent environments."""
    env_raw.target = target.copy().astype(np.float32)
    env_raw.ball = ball.copy().astype(np.float32)
    if hasattr(env_raw, 'theta3'):
        eff = env_raw.forward_kinematics(env_raw.theta1, env_raw.theta2, env_raw.theta3)
    else:
        eff = env_raw.forward_kinematics(env_raw.theta1, env_raw.theta2)
    env_raw.prev_dist_ball_target = float(np.linalg.norm(ball - target))
    env_raw.prev_dist_eff_ball = float(np.linalg.norm(eff - ball))


def generate_independent_configs(n_configs, rng, max_reach):
    """Generate target and ball positions."""
    configs = []
    for _ in range(n_configs):
        # Target
        r_t = rng.uniform(0.4 * max_reach, 0.75 * max_reach)
        a_t = rng.uniform(-np.pi, np.pi)
        target = np.array([r_t * np.cos(a_t), r_t * np.sin(a_t)])

        # Ball at least 0.3m from target
        for _ in range(1000):
            r_b = rng.uniform(0.3 * max_reach, 0.75 * max_reach)
            a_b = rng.uniform(-np.pi, np.pi)
            ball = np.array([r_b * np.cos(a_b), r_b * np.sin(a_b)])
            if np.linalg.norm(ball - target) >= 0.3:
                break
        else:
            ball = target + np.array([0.4, 0.0])

        configs.append((target, ball))
    return configs


def generate_one_pair(policy_2dof, vn_2dof, policy_3dof, vn_3dof,
                      pair_idx, n_configs, steps_per_config,
                      fixed_seed, max_reach):
    seed = fixed_seed + pair_idx
    rng = np.random.RandomState(seed)

    env_2dof = PushBallEnv_2dof(render_mode=None, max_steps=1_000_000)
    env_3dof = PushBallEnv_3dof(render_mode=None, max_steps=1_000_000)

    obs_2dof, _ = env_2dof.reset(seed=seed)
    obs_3dof, _ = env_3dof.reset(seed=seed)

    configs = generate_independent_configs(n_configs, rng, max_reach)

    # Set first config
    target, ball = configs[0]
    override_target_and_ball(env_2dof, target, ball)
    override_target_and_ball(env_3dof, target, ball)

    obs_2dof = env_2dof._get_obs()
    obs_3dof = env_3dof._get_obs()

    states_2dof = []
    states_3dof = []
    actions_2dof = []
    actions_3dof = []

    config_idx = 0
    step_in_config = 0
    first_config_steps = 30

    total_steps = first_config_steps + (n_configs - 1) * steps_per_config

    for _ in range(total_steps):
        if config_idx == 0:
            if step_in_config >= first_config_steps and config_idx < n_configs - 1:
                config_idx += 1
                step_in_config = 0
                target, ball = configs[config_idx]
                override_target_and_ball(env_2dof, target, ball)
                override_target_and_ball(env_3dof, target, ball)
        else:
            if step_in_config >= steps_per_config and config_idx < n_configs - 1:
                config_idx += 1
                step_in_config = 0
                target, ball = configs[config_idx]
                override_target_and_ball(env_2dof, target, ball)
                override_target_and_ball(env_3dof, target, ball)

        obs_2dof = env_2dof._get_obs()
        obs_3dof = env_3dof._get_obs()

        act_2dof = predict_action(policy_2dof, vn_2dof, obs_2dof)
        act_3dof = predict_action(policy_3dof, vn_3dof, obs_3dof)

        states_2dof.append(obs_2dof.copy())  # Full 10D
        states_3dof.append(obs_3dof.copy())  # Full 12D
        actions_2dof.append(act_2dof.copy())
        actions_3dof.append(act_3dof.copy())

        obs_2dof, _, _, _, _ = env_2dof.step(act_2dof)
        obs_3dof, _, _, _, _ = env_3dof.step(act_3dof)

        step_in_config += 1

    env_2dof.close()
    env_3dof.close()

    return {
        'states_2dof': np.array(states_2dof, dtype=np.float32),
        'states_3dof': np.array(states_3dof, dtype=np.float32),
        'actions_2dof': np.array(actions_2dof, dtype=np.float32),
        'actions_3dof': np.array(actions_3dof, dtype=np.float32),
        'configs': configs,
        'metadata': {
            'pair_idx': pair_idx, 'seed': seed,
            'n_steps': len(states_2dof),
            'source': 'ppo_policy',
        }
    }


def main():
    RUN_ID_2DOF = 1
    RUN_ID_3DOF = 1

    MODEL_2DOF = f"./models/ppo_pushball_2dof_{RUN_ID_2DOF}/best_model.zip"
    VECNORM_2DOF = f"./models/ppo_pushball_2dof_{RUN_ID_2DOF}/vec_normalize.pkl"
    MODEL_3DOF = f"./models/ppo_pushball_3dof_{RUN_ID_3DOF}/best_model.zip"
    VECNORM_3DOF = f"./models/ppo_pushball_3dof_{RUN_ID_3DOF}/vec_normalize.pkl"

    N_PAIRS = 2000
    N_CONFIGS = 100
    STEPS_PER_CONFIG = 15
    FIRST_CONFIG_STEPS = 30
    FIXED_SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_DIR = Path("./data/LSUNN")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAJ_PATH = DATA_DIR / "trajectories_pushball.pkl"

    MAX_REACH = min(
        PushBallEnv_2dof().max_reach,
        PushBallEnv_3dof().max_reach
    )

    def make_2dof(): return Monitor(PushBallEnv_2dof(render_mode=None))
    def make_3dof(): return Monitor(PushBallEnv_3dof(render_mode=None))

    policy_2dof, vn_2dof, venv_2dof = load_policy_and_vecnorm(
        MODEL_2DOF, VECNORM_2DOF, make_2dof, DEVICE
    )
    policy_3dof, vn_3dof, venv_3dof = load_policy_and_vecnorm(
        MODEL_3DOF, VECNORM_3DOF, make_3dof, DEVICE
    )

    all_s2, all_s3, all_a2, all_a3, all_cfg = [], [], [], [], []

    steps_per_pair = FIRST_CONFIG_STEPS + (N_CONFIGS - 1) * STEPS_PER_CONFIG

    print(f"\nGenerating {N_PAIRS * steps_per_pair:,} paired samples...")

    pbar = tqdm(range(N_PAIRS), desc="Trajectory pairs", unit="pair")
    for i in pbar:
        pair = generate_one_pair(
            policy_2dof, vn_2dof, policy_3dof, vn_3dof,
            pair_idx=i, n_configs=N_CONFIGS,
            steps_per_config=STEPS_PER_CONFIG,
            fixed_seed=FIXED_SEED, max_reach=MAX_REACH,
        )
        all_s2.append(pair['states_2dof'])
        all_s3.append(pair['states_3dof'])
        all_a2.append(pair['actions_2dof'])
        all_a3.append(pair['actions_3dof'])
        all_cfg.append(pair['configs'])
        pbar.set_postfix({"samples": f"{(i+1)*steps_per_pair:,}"})

    trajectories = {
        'states_2dof': all_s2,
        'states_3dof': all_s3,
        'actions_2dof': all_a2,
        'actions_3dof': all_a3,
        'configs': all_cfg,
        'metadata': {
            'n_pairs': N_PAIRS, 'seed': FIXED_SEED,
            'arm_size_2dof': 6, 'arm_size_3dof': 8,
            'source': 'ppo_policy',
            'steps_per_config': STEPS_PER_CONFIG,
        }
    }

    with open(TRAJ_PATH, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"\n✓ Saved → {TRAJ_PATH}")
    venv_2dof.close()
    venv_3dof.close()


if __name__ == "__main__":
    main()
