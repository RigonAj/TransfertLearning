import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lsunn.latent_envs import make_latent_pushball_env
from lsunn.mapper_models import LATENT_PUSHBALL_DIM


def train_pushball_latent(
    source_domain: str,
    mappers_dir: str,
    save_dir: str,
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    device: str = "cpu",
    use_subproc: bool = False,
):
    if source_domain not in {"2dof", "3dof"}:
        raise ValueError("source_domain must be '2dof' or '3dof'")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    env_fn = make_latent_pushball_env(
        policy_domain=source_domain,
        raw_domain=source_domain,
        mappers_dir=mappers_dir,
        max_steps=150,
        render_mode=None,
        device=device,
    )

    if use_subproc:
        train_env = SubprocVecEnv([env_fn for _ in range(n_envs)])
    else:
        train_env = DummyVecEnv([env_fn for _ in range(n_envs)])

    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    ppo = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
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
        tensorboard_log=str(save_path / "tb"),
    )

    ppo.learn(total_timesteps=total_timesteps, progress_bar=True)

    ppo.save(save_path / f"pushball_policy_{source_domain}")
    train_env.save(save_path / "vec_normalize.pkl")
    train_env.close()
    print(f"Pushball latent policy saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-domain", choices=["2dof", "3dof"], default="2dof")
    parser.add_argument("--mappers-dir", default="./data/LSUNN/latent_reaching")
    parser.add_argument("--save-dir", default="./data/LSUNN/latent_pushball_2dof")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-subproc", action="store_true")
    args = parser.parse_args()

    if args.save_dir == "./data/LSUNN/latent_pushball_2dof" and args.source_domain == "3dof":
        args.save_dir = "./data/LSUNN/latent_pushball_3dof"

    train_pushball_latent(**vars(args))


if __name__ == "__main__":
    main()
