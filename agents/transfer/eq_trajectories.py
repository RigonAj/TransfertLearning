import os
import pickle
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_continuous_reaching_2dof import Arm2DoFPersistentEnv
from envs.env_continuous_reaching_3dof import Arm3DoFPersistentEnv

import torch
torch.set_num_threads(4)


# =========================================================
# Helpers
# =========================================================

def load_policy_and_vecnorm(model_path: str, vec_path: str, env_factory, device="cpu"):
    model = PPO.load(model_path, device=device)
    venv = DummyVecEnv([env_factory])
    vec_norm = VecNormalize.load(vec_path, venv=venv)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return model, vec_norm, venv


def predict_action(model, vec_norm, raw_obs: np.ndarray) -> np.ndarray:
    obs_norm = vec_norm.normalize_obs(raw_obs.reshape(1, -1))
    action, _ = model.predict(obs_norm, deterministic=True)  #####
    return action[0]


def override_target(env_raw, target: np.ndarray):
    env_raw.target = target.copy()
    if hasattr(env_raw, 'theta3'):
        eff = env_raw.forward_kinematics(env_raw.theta1, env_raw.theta2, env_raw.theta3)
    else:
        eff = env_raw.forward_kinematics(env_raw.theta1, env_raw.theta2)
    env_raw.prev_dist = float(np.linalg.norm(eff - target))


# =========================================================
# Génération de cibles : indépendantes et uniformes
# =========================================================

def generate_independent_targets(
    n_targets: int,
    rng: np.random.RandomState,
    max_reach: float,
) -> np.ndarray:
    """
    Génère n_targets cibles 2D indépendantes, uniformément réparties dans le disque.
    """
    targets = []
    for _ in range(n_targets):
        # rayon uniforme entre 0 et 0.92*max_reach (pour éviter les butées strictes)
        r = rng.uniform(0.0, max_reach * 0.92)
        angle = rng.uniform(-np.pi, np.pi)
        targets.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(targets, dtype=np.float32)


# =========================================================
# Génération d'une paire de trajectoires alignées
# =========================================================

def generate_one_pair(
    policy_2dof, vec_norm_2dof,
    policy_3dof, vec_norm_3dof,
    pair_idx: int,
    n_targets: int,
    steps_per_target: int,
    fixed_seed: int,
    max_reach: float,
    first_target_steps: int = 50,
) -> dict:

    seed = fixed_seed + pair_idx
    rng = np.random.RandomState(seed)

    env_2dof = Arm2DoFPersistentEnv(render_mode=None)
    env_3dof = Arm3DoFPersistentEnv(render_mode=None)

    env_2dof.max_steps = 1_000_000
    env_3dof.max_steps = 1_000_000

    arm_size_2 = env_2dof.arm_obs_size
    arm_size_3 = env_3dof.arm_obs_size

    obs_2dof, _ = env_2dof.reset(seed=seed)
    obs_3dof, _ = env_3dof.reset(seed=seed)

    # Génération de cibles indépendantes
    targets = generate_independent_targets(n_targets, rng, max_reach)

    # Définit la première cible
    override_target(env_2dof, targets[0])
    override_target(env_3dof, targets[0])
    obs_2dof = env_2dof._get_obs()
    obs_3dof = env_3dof._get_obs()

    states_2dof, states_3dof = [], []
    actions_2dof, actions_3dof = [], []

    target_idx = 0
    step_in_target = 0
    total_steps = first_target_steps + (n_targets - 1) * steps_per_target

    for _ in range(total_steps):
        # Changement de cible
        if target_idx == 0:
            if step_in_target >= first_target_steps and target_idx < n_targets - 1:
                target_idx += 1
                step_in_target = 0
        else:
            if step_in_target >= steps_per_target and target_idx < n_targets - 1:
                target_idx += 1
                step_in_target = 0

        override_target(env_2dof, targets[target_idx])
        override_target(env_3dof, targets[target_idx])

        obs_2dof = env_2dof._get_obs()
        obs_3dof = env_3dof._get_obs()

        act_2dof = predict_action(policy_2dof, vec_norm_2dof, obs_2dof)
        act_3dof = predict_action(policy_3dof, vec_norm_3dof, obs_3dof)

        states_2dof.append(obs_2dof[:arm_size_2].copy())
        states_3dof.append(obs_3dof[:arm_size_3].copy())
        actions_2dof.append(act_2dof.copy())
        actions_3dof.append(act_3dof.copy())

        obs_2dof, _, _, _, _ = env_2dof.step(act_2dof)
        obs_3dof, _, _, _, _ = env_3dof.step(act_3dof)

        step_in_target += 1

    env_2dof.close()
    env_3dof.close()

    return {
        'states_2dof':  np.array(states_2dof,  dtype=np.float32),
        'states_3dof':  np.array(states_3dof,  dtype=np.float32),
        'actions_2dof': np.array(actions_2dof, dtype=np.float32),
        'actions_3dof': np.array(actions_3dof, dtype=np.float32),
        'targets': targets,
        'metadata': {
            'pair_idx':    pair_idx,
            'seed':        seed,
            'n_steps':     len(states_2dof),
            'arm_size_2':  arm_size_2,
            'arm_size_3':  arm_size_3,
            'source':      'ppo_policy',
        }
    }


# =========================================================
# Main
# =========================================================

def main():
    run_id_2dof = 1
    run_id_3dof = 1

    MODEL_2DOF   = f"./models/ppo_reach_2dof_{run_id_2dof}/best_model.zip"
    VECNORM_2DOF = f"./models/ppo_reach_2dof_{run_id_2dof}/vec_normalize.pkl"
    MODEL_3DOF   = f"./models/ppo_reach_3dof_{run_id_3dof}/best_model.zip"
    VECNORM_3DOF = f"./models/ppo_reach_3dof_{run_id_3dof}/vec_normalize.pkl"

    #####
    N_PAIRS            = 2000
    N_TARGETS          = 500
    STEPS_PER_TARGET   = 15
    FIRST_TARGET_STEPS = 30

    n_samples = N_PAIRS * (FIRST_TARGET_STEPS + (N_TARGETS - 1) * STEPS_PER_TARGET)
    print("Génération de ", n_samples, " échantillons...")

    MAX_REACH = min(
        Arm2DoFPersistentEnv().max_reach,
        Arm3DoFPersistentEnv().max_reach
    )

    FIXED_SEED = 42
    DEVICE = "cpu"

    data_dir = Path("./data/transfer_learning")
    data_dir.mkdir(parents=True, exist_ok=True)
    traj_path = data_dir / "trajectories.pkl"

    def make_2dof(): return Monitor(Arm2DoFPersistentEnv(render_mode=None))
    def make_3dof(): return Monitor(Arm3DoFPersistentEnv(render_mode=None))

    policy_2dof, vec_norm_2dof, venv_2dof = load_policy_and_vecnorm(
        MODEL_2DOF, VECNORM_2DOF, make_2dof, DEVICE)
    policy_3dof, vec_norm_3dof, venv_3dof = load_policy_and_vecnorm(
        MODEL_3DOF, VECNORM_3DOF, make_3dof, DEVICE)

    all_s2, all_s3, all_a2, all_a3, all_tgt = [], [], [], [], []

    for i in range(N_PAIRS):
        pair = generate_one_pair(
            policy_2dof, vec_norm_2dof,
            policy_3dof, vec_norm_3dof,
            pair_idx=i,
            n_targets=N_TARGETS,
            steps_per_target=STEPS_PER_TARGET,
            fixed_seed=FIXED_SEED,
            max_reach=MAX_REACH,
            first_target_steps=FIRST_TARGET_STEPS,
        )

        all_s2.append(pair['states_2dof'])
        all_s3.append(pair['states_3dof'])
        all_a2.append(pair['actions_2dof'])
        all_a3.append(pair['actions_3dof'])
        all_tgt.append(pair['targets'])

    trajectories = {
        'states_2dof':  all_s2,
        'states_3dof':  all_s3,
        'actions_2dof': all_a2,
        'actions_3dof': all_a3,
        'targets':      all_tgt,
        'metadata': {
            'n_pairs': N_PAIRS,
            'seed': FIXED_SEED,
            'arm_size_2dof': Arm2DoFPersistentEnv().arm_obs_size,
            'arm_size_3dof': Arm3DoFPersistentEnv().arm_obs_size,
            'source': 'ppo_policy',
            'steps_per_target': STEPS_PER_TARGET,
        }
    }

    with open(traj_path, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"\n✓ Saved trajectories → {traj_path}")

    venv_2dof.close()
    venv_3dof.close()


if __name__ == "__main__":
    main()
