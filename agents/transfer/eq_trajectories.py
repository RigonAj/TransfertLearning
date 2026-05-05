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

# ============================================================================
# NOTE: Les observations sont enregistrées comme des "arm_obs" uniquement,
# c'est-à-dire le bloc bras normalisé hérité de Arm2DoF / Arm3DoF :
#   - 2-DoF : obs[:6]  → [θ1/π, θ2/π, dθ1/ω, dθ2/ω, eff_x/r, eff_y/r]
#   - 3-DoF : obs[:8]  → [θ1/π, θ2/π, θ3/π, dθ1/ω, dθ2/ω, dθ3/ω, eff_x/r, eff_y/r]
# Les dimensions tâche (target, dx, dy, dist) ne sont PAS enregistrées.
# Cela permet d'entraîner des mappers bras↔bras indépendants de la tâche.
# ============================================================================

# ============================================================================
# Helpers
# ============================================================================

def load_policy_and_vecnorm(model_path: str, vec_path: str, env_factory, device="cpu"):
    """Load a PPO model + its VecNormalize (inference mode)."""
    model = PPO.load(model_path, device=device)
    venv = DummyVecEnv([env_factory])
    vec_norm = VecNormalize.load(vec_path, venv=venv)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return model, vec_norm, venv


def predict_action(model, vec_norm, raw_obs: np.ndarray) -> np.ndarray:
    """Normalize a raw single observation and return a deterministic action."""
    obs_norm = vec_norm.normalize_obs(raw_obs.reshape(1, -1))
    action, _ = model.predict(obs_norm, deterministic=True)
    return action[0]  # shape (n_actions,)


def override_target(env_raw, target: np.ndarray):
    """
    Inject a target directly into the raw Gym env and update prev_dist.
    Works because ReachingEnv_2dof / ReachingEnv_3dof expose .target et .prev_dist.
    """
    env_raw.target = target.copy()
    if hasattr(env_raw, 'theta3'):
        eff = env_raw.forward_kinematics(env_raw.theta1, env_raw.theta2, env_raw.theta3)
    else:
        eff = env_raw.forward_kinematics(env_raw.theta1, env_raw.theta2)
    env_raw.prev_dist = float(np.linalg.norm(eff - target))


# ============================================================================
# Trajectory generation
# ============================================================================

def generate_target_sequence(
    start_pos: np.ndarray,
    n_targets: int,
    target_distance: float,
    rng: np.random.RandomState,
    max_reach: float,
) -> np.ndarray:
    """
    Chain of targets, each `target_distance` away from the previous one.
    Targets are clipped to the reachable workspace (circle of radius max_reach).
    """
    targets = []
    current = start_pos.copy()
    for _ in range(n_targets):
        angle = rng.uniform(0, 2 * np.pi)
        nxt = current + target_distance * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        r = np.linalg.norm(nxt)
        if r > max_reach * 0.92:
            nxt = nxt / r * max_reach * 0.92
        targets.append(nxt)
        current = nxt
    return np.stack(targets)  # (n_targets, 2)


def generate_one_pair(
    policy_2dof, vec_norm_2dof,
    policy_3dof, vec_norm_3dof,
    pair_idx: int,
    n_targets: int,
    steps_per_target: int,
    fixed_seed: int,
    target_distance: float,
    max_reach: float,
    first_target_steps: int = 50,
) -> dict:
    """
    One aligned trajectory pair WITHOUT ANY RESET.
    Both arms follow the same target sequence until the end.
    """
    seed = fixed_seed + pair_idx
    rng = np.random.RandomState(seed)

    # --- Create raw envs ---
    env_2dof = Arm2DoFPersistentEnv(render_mode=None)
    env_3dof = Arm3DoFPersistentEnv(render_mode=None)

    # Disable time truncation (max_steps)
    env_2dof.max_steps = 1_000_000
    env_3dof.max_steps = 1_000_000

    arm_size_2 = env_2dof.arm_obs_size   # 6
    arm_size_3 = env_3dof.arm_obs_size   # 8

    # Reset with same seed → same initial joint angles for both arms
    obs_2dof, _ = env_2dof.reset(seed=seed)
    obs_3dof, _ = env_3dof.reset(seed=seed)

    # Initial end‑effector position (from 2‑DoF, used to start the target sequence)
    eff_init = obs_2dof[4:6] * max_reach

    # Generate common target sequence
    targets = generate_target_sequence(eff_init, n_targets, target_distance, rng, max_reach)

    # Inject first target into both environments
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
        # --- Target switching logic (same as before) ---
        if target_idx == 0:
            if step_in_target >= first_target_steps and target_idx < n_targets - 1:
                target_idx += 1
                step_in_target = 0
                override_target(env_2dof, targets[target_idx])
                override_target(env_3dof, targets[target_idx])
                obs_2dof = env_2dof._get_obs()
                obs_3dof = env_3dof._get_obs()
        else:
            if step_in_target >= steps_per_target and target_idx < n_targets - 1:
                target_idx += 1
                step_in_target = 0
                override_target(env_2dof, targets[target_idx])
                override_target(env_3dof, targets[target_idx])
                obs_2dof = env_2dof._get_obs()
                obs_3dof = env_3dof._get_obs()

        # Get actions from trained policies
        act_2dof = predict_action(policy_2dof, vec_norm_2dof, obs_2dof)
        act_3dof = predict_action(policy_3dof, vec_norm_3dof, obs_3dof)

        # Record arm_obs only
        states_2dof.append(obs_2dof[:arm_size_2].copy())
        states_3dof.append(obs_3dof[:arm_size_3].copy())
        actions_2dof.append(act_2dof.copy())
        actions_3dof.append(act_3dof.copy())

        # Step environments – ignore terminated/truncated flags
        obs_2dof, _, _, _, _ = env_2dof.step(act_2dof)
        obs_3dof, _, _, _, _ = env_3dof.step(act_3dof)

        step_in_target += 1

    # No reset at the end – we keep the environments as they are
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


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    # ----- Configuration -----
    run_id_2dof = 1
    run_id_3dof = 1
    MODEL_2DOF   = f"./models/ppo_reach_2dof_{run_id_2dof}/best_model.zip"
    VECNORM_2DOF = f"./models/ppo_reach_2dof_{run_id_2dof}/vec_normalize.pkl"
    MODEL_3DOF   = f"./models/ppo_reach_3dof_{run_id_3dof}/best_model.zip"
    VECNORM_3DOF = f"./models/ppo_reach_3dof_{run_id_3dof}/vec_normalize.pkl"

    N_PAIRS            = 200    # more data → better mapper generalisation
    N_TARGETS          = 100
    STEPS_PER_TARGET   = 12     # longer windows → more diverse state coverage
    FIRST_TARGET_STEPS = 50     # first target stays in place for 50 steps
    TARGET_DISTANCE    = 0.3    # larger step → more workspace coverage
#    MAX_REACH          = 3.0 ## 
    MAX_REACH 		= min(Arm2DoFPersistentEnv().max_reach, Arm3DoFPersistentEnv().max_reach)
    FIXED_SEED         = 42
    DEVICE             = "cpu"

    data_dir  = Path("./data/transfer_learning")
    data_dir.mkdir(parents=True, exist_ok=True)
    traj_path = data_dir / "trajectories.pkl"

    print("\n" + "="*70)
    print("PART 1: TRAJECTORY GENERATION (PPO-POLICY BASED)")
    print("="*70)
    print("\n  Observations enregistrées : bloc bras uniquement")
    print("    2-DoF arm_obs : 6D  [θ1/π, θ2/π, dθ1/ω, dθ2/ω, eff_x/r, eff_y/r]")
    print("    3-DoF arm_obs : 8D  [θ1/π, θ2/π, θ3/π, dθ1/ω, dθ2/ω, dθ3/ω, eff_x/r, eff_y/r]")

    # ----- Load policies -----
    def make_2dof(): return Monitor(Arm2DoFPersistentEnv(render_mode=None))
    def make_3dof(): return Monitor(Arm3DoFPersistentEnv(render_mode=None))

    print("\n[1] Loading trained PPO policies...")
    policy_2dof, vec_norm_2dof, venv_2dof = load_policy_and_vecnorm(
        MODEL_2DOF, VECNORM_2DOF, make_2dof, DEVICE)
    policy_3dof, vec_norm_3dof, venv_3dof = load_policy_and_vecnorm(
        MODEL_3DOF, VECNORM_3DOF, make_3dof, DEVICE)
    print("  ✓ 2-DoF policy loaded")
    print("  ✓ 3-DoF policy loaded")

    # ----- Generate pairs -----
    total_steps_per_pair = FIRST_TARGET_STEPS + (N_TARGETS - 1) * STEPS_PER_TARGET
    print(f"\n[2] Generating {N_PAIRS} pairs × {total_steps_per_pair} steps = "
          f"{N_PAIRS * total_steps_per_pair:,} total steps\n")
    print(f"    (First target: {FIRST_TARGET_STEPS} steps, then {N_TARGETS-1} targets × {STEPS_PER_TARGET} steps)\n")

    all_s2, all_s3, all_a2, all_a3, all_tgt = [], [], [], [], []

    for i in range(N_PAIRS):
        pair = generate_one_pair(
            policy_2dof, vec_norm_2dof,
            policy_3dof, vec_norm_3dof,
            pair_idx=i,
            n_targets=N_TARGETS,
            steps_per_target=STEPS_PER_TARGET,
            fixed_seed=FIXED_SEED,
            target_distance=TARGET_DISTANCE,
            max_reach=MAX_REACH,
            first_target_steps=FIRST_TARGET_STEPS,
        )
        all_s2.append(pair['states_2dof'])
        all_s3.append(pair['states_3dof'])
        all_a2.append(pair['actions_2dof'])
        all_a3.append(pair['actions_3dof'])
        all_tgt.append(pair['targets'])

        if i == 1:
            print("Starting pairs generation")
        if (i + 1) % 25 == 0:
            n = sum(x.shape[0] for x in all_s2)
            print(f"  [{i + 1:3d}/{N_PAIRS}] pairs generated — {n:,} steps total")

    trajectories = {
        'states_2dof':  all_s2,   # list of (T, 6) arrays
        'states_3dof':  all_s3,   # list of (T, 8) arrays
        'actions_2dof': all_a2,   # list of (T, 2) arrays
        'actions_3dof': all_a3,   # list of (T, 3) arrays
        'targets':      all_tgt,
        'metadata': {
            'n_pairs':            N_PAIRS,
            'n_targets':          N_TARGETS,
            'steps_per_target':   STEPS_PER_TARGET,
            'first_target_steps': FIRST_TARGET_STEPS,
            'target_distance':    TARGET_DISTANCE,
            'seed':               FIXED_SEED,
            'arm_size_2dof':      6,
            'arm_size_3dof':      8,
            'source':             'ppo_policy',
        }
    }

    with open(traj_path, 'wb') as f:
        pickle.dump(trajectories, f)

    n_total = sum(x.shape[0] for x in all_s2)
    print(f"\n✓ Saved {N_PAIRS} trajectory pairs ({n_total:,} steps) → {traj_path}")
    print("  states_2dof shape per pair :", all_s2[0].shape, " (arm_obs 6D)")
    print("  states_3dof shape per pair :", all_s3[0].shape, " (arm_obs 8D)")
    print("\nNext step: run eq_mappings_2to3dof.py or eq_mappings_3to2dof.py")
    print("="*70 + "\n")

    venv_2dof.close()
    venv_3dof.close()


if __name__ == "__main__":
    main()
