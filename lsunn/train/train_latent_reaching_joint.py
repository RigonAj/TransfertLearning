import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.env_reaching_2dof import ReachingEnv_2dof
from envs.env_reaching_3dof import ReachingEnv_3dof
from lsunn.mapper_models import (
    ACTION_DIM_2DOF,
    ACTION_DIM_3DOF,
    ARM_OBS_2DOF,
    ARM_OBS_3DOF,
    ActionMapperMLP,
    StateMapperMLP,
    angular_mse,
    fk_from_arm_state_torch,
    save_latent_mappers,
)


def make_reaching_env_2dof(seed: int = 0):
    def _init():
        env = ReachingEnv_2dof(render_mode=None)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def make_reaching_env_3dof(seed: int = 0):
    def _init():
        env = ReachingEnv_3dof(render_mode=None)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def build_ppo(vec_env, device: str, n_steps: int = 512):
    return PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
        verbose=1,
    )


def collect_paired_reaching_batch(
    n_pairs: int,
    seq_len: int,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    env2 = ReachingEnv_2dof(render_mode=None)
    env3 = ReachingEnv_3dof(render_mode=None)

    states2 = []
    states3 = []
    actions2 = []
    actions3 = []

    for _ in tqdm(range(n_pairs), desc="paired reaching samples"):
        seed = int(rng.randint(0, 2**31 - 1))
        obs2, _ = env2.reset(seed=seed)
        obs3, _ = env3.reset(seed=seed)
        env3.target = env2.target.copy()
        eff3 = env3.get_end_effector_pos()
        env3.prev_dist = float(np.linalg.norm(eff3 - env3.target))

        for _ in range(seq_len):
            a2 = rng.uniform(-1.0, 1.0, ACTION_DIM_2DOF).astype(np.float32)
            a3 = rng.uniform(-1.0, 1.0, ACTION_DIM_3DOF).astype(np.float32)

            obs2, _, terminated2, truncated2, _ = env2.step(a2)
            obs3, _, terminated3, truncated3, _ = env3.step(a3)

            states2.append(obs2[:ARM_OBS_2DOF].astype(np.float32))
            states3.append(obs3[:ARM_OBS_3DOF].astype(np.float32))
            actions2.append(a2.astype(np.float32))
            actions3.append(a3.astype(np.float32))

            if terminated2 or truncated2 or terminated3 or truncated3:
                break

    env2.close()
    env3.close()

    return {
        "s2": np.asarray(states2, dtype=np.float32),
        "s3": np.asarray(states3, dtype=np.float32),
        "a2": np.asarray(actions2, dtype=np.float32),
        "a3": np.asarray(actions3, dtype=np.float32),
    }


def mapper_losses(
    state_2to3: StateMapperMLP,
    state_3to2: StateMapperMLP,
    action_2to3: ActionMapperMLP,
    action_3to2: ActionMapperMLP,
    s2: torch.Tensor,
    s3: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    pred_s3 = state_2to3(s2)
    pred_s2 = state_3to2(s3)

    s2_rec = state_3to2(pred_s3)
    s3_rec = state_2to3(pred_s2)

    mse = nn.MSELoss()
    state_forward = mse(pred_s3, s3) + mse(pred_s2, s2)
    state_cycle = mse(s2_rec, s2) + mse(s3_rec, s3)
    state_fk = mse(fk_from_arm_state_torch(pred_s3), fk_from_arm_state_torch(s3)) + mse(
        fk_from_arm_state_torch(pred_s2), fk_from_arm_state_torch(s2)
    )
    state_loss = (
        weights["state_forward"] * state_forward
        + weights["state_cycle"] * state_cycle
        + weights["fk"] * state_fk
    )

    pred_a3 = action_2to3(s3, a2)
    pred_a2 = action_3to2(s2, a3)

    a2_rec = action_3to2(s2, pred_a3)
    a3_rec = action_2to3(s3, pred_a2)

    action_forward = mse(pred_a3, a3) + mse(pred_a2, a2)
    action_cycle = mse(a2_rec, a2) + mse(a3_rec, a3)
    action_loss = weights["action_forward"] * action_forward + weights["action_cycle"] * action_cycle

    return state_loss, action_loss, {
        "state_forward": float(state_forward.detach().cpu()),
        "state_cycle": float(state_cycle.detach().cpu()),
        "state_fk": float(state_fk.detach().cpu()),
        "action_forward": float(action_forward.detach().cpu()),
        "action_cycle": float(action_cycle.detach().cpu()),
    }


def train_mappers_on_batch(
    state_2to3: StateMapperMLP,
    state_3to2: StateMapperMLP,
    action_2to3: ActionMapperMLP,
    action_3to2: ActionMapperMLP,
    batch: Dict[str, np.ndarray],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weights: Dict[str, float],
) -> Dict[str, float]:
    s2 = torch.tensor(batch["s2"], dtype=torch.float32, device=device)
    s3 = torch.tensor(batch["s3"], dtype=torch.float32, device=device)
    a2 = torch.tensor(batch["a2"], dtype=torch.float32, device=device)
    a3 = torch.tensor(batch["a3"], dtype=torch.float32, device=device)

    state_opt = optim.Adam(
        list(state_2to3.parameters()) + list(state_3to2.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
    action_opt = optim.Adam(
        list(action_2to3.parameters()) + list(action_3to2.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )

    n = len(s2)
    last_metrics = {}
    for _ in tqdm(range(epochs), desc="mapper epochs"):
        perm = torch.randperm(n, device=device)
        total_state = 0.0
        total_action = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            b_s2 = s2[idx]
            b_s3 = s3[idx]
            b_a2 = a2[idx]
            b_a3 = a3[idx]

            state_loss, action_loss, metrics = mapper_losses(
                state_2to3,
                state_3to2,
                action_2to3,
                action_3to2,
                b_s2,
                b_s3,
                b_a2,
                b_a3,
                weights,
            )

            state_opt.zero_grad()
            state_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(state_2to3.parameters()) + list(state_3to2.parameters()),
                1.0,
            )
            state_opt.step()

            action_opt.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(action_2to3.parameters()) + list(action_3to2.parameters()),
                1.0,
            )
            action_opt.step()

            total_state += float(state_loss.detach().cpu())
            total_action += float(action_loss.detach().cpu())
            n_batches += 1
            last_metrics = metrics

    return {
        "state_loss": total_state / max(1, n_batches),
        "action_loss": total_action / max(1, n_batches),
        **last_metrics,
    }


def train_reaching_latent_joint(
    save_dir: str,
    total_timesteps: int = 1_000_000,
    mapper_update_timesteps: int = 50_000,
    mapper_pairs: int = 4096,
    mapper_seq_len: int = 20,
    mapper_epochs: int = 4,
    mapper_batch_size: int = 512,
    mapper_lr: float = 1e-3,
    device: str = "cpu",
    hidden: int = 512,
):
    torch.set_num_threads(8)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    state_2to3 = StateMapperMLP(ARM_OBS_2DOF, ARM_OBS_3DOF, hidden=hidden).to(device)
    state_3to2 = StateMapperMLP(ARM_OBS_3DOF, ARM_OBS_2DOF, hidden=hidden).to(device)
    action_2to3 = ActionMapperMLP(ARM_OBS_3DOF, ACTION_DIM_2DOF, ACTION_DIM_3DOF, hidden=hidden).to(device)
    action_3to2 = ActionMapperMLP(ARM_OBS_2DOF, ACTION_DIM_3DOF, ACTION_DIM_2DOF, hidden=hidden).to(device)

    vec_env2 = DummyVecEnv([make_reaching_env_2dof(i) for i in range(16)])
    vec_env3 = DummyVecEnv([make_reaching_env_3dof(i) for i in range(16)])

    vec_env2 = VecNormalize(vec_env2, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    vec_env3 = VecNormalize(vec_env3, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    ppo2 = build_ppo(vec_env2, device=device)
    ppo3 = build_ppo(vec_env3, device=device)

    weights = {
        "state_forward": 4.0,
        "state_cycle": 2.0,
        "fk": 0.5,
        "action_forward": 4.0,
        "action_cycle": 2.0,
    }

    rng = np.random.RandomState(42)
    learned = 0
    update = 0
    while learned < total_timesteps:
        steps = min(mapper_update_timesteps, total_timesteps - learned)
        update += 1

        print(f"\n=== Reaching PPO update {update} | learned={learned}/{total_timesteps} ===")
        ppo2.learn(total_timesteps=steps, reset_num_timesteps=(update == 1), progress_bar=True)
        ppo3.learn(total_timesteps=steps, reset_num_timesteps=(update == 1), progress_bar=True)
        learned += steps

        batch = collect_paired_reaching_batch(
            n_pairs=mapper_pairs,
            seq_len=mapper_seq_len,
            rng=rng,
        )
        metrics = train_mappers_on_batch(
            state_2to3,
            state_3to2,
            action_2to3,
            action_3to2,
            batch,
            device=device,
            epochs=mapper_epochs,
            batch_size=mapper_batch_size,
            lr=mapper_lr,
            weights=weights,
        )
        print("Mapper losses:", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        if update % 5 == 0 or learned >= total_timesteps:
            ppo2.save(save_path / "reaching_policy_2dof")
            ppo3.save(save_path / "reaching_policy_3dof")
            vec_env2.save(save_path / "vec_normalize_reaching_2dof.pkl")
            vec_env3.save(save_path / "vec_normalize_reaching_3dof.pkl")
            save_latent_mappers(
                save_path,
                state_2to3,
                state_3to2,
                action_2to3,
                action_3to2,
                run_id="latent_reaching",
            )

    ppo2.save(save_path / "reaching_policy_2dof_final")
    ppo3.save(save_path / "reaching_policy_3dof_final")
    vec_env2.save(save_path / "vec_normalize_reaching_2dof_final.pkl")
    vec_env3.save(save_path / "vec_normalize_reaching_3dof_final.pkl")
    save_latent_mappers(
        save_path,
        state_2to3,
        state_3to2,
        action_2to3,
        action_3to2,
        run_id="latent_reaching",
    )

    vec_env2.close()
    vec_env3.close()
    print(f"Latent reaching model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="./data/LSUNN/latent_reaching")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--mapper-update-timesteps", type=int, default=50_000)
    parser.add_argument("--mapper-pairs", type=int, default=4096)
    parser.add_argument("--mapper-seq-len", type=int, default=20)
    parser.add_argument("--mapper-epochs", type=int, default=4)
    parser.add_argument("--mapper-batch-size", type=int, default=512)
    parser.add_argument("--mapper-lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_reaching_latent_joint(**vars(args))


if __name__ == "__main__":
    main()
