import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof
from lsunn.latent_envs import make_latent_pushball_env
from lsunn.mapper_models import load_latent_mappers


def transfer_pushball_latent(
    source_domain: str,
    target_domain: str,
    mappers_dir: str,
    policy_dir: str,
    n_episodes: int = 100,
    max_steps: int = 150,
    device: str = "cpu",
):
    if source_domain not in {"2dof", "3dof"} or target_domain not in {"2dof", "3dof"}:
        raise ValueError("source_domain and target_domain must be '2dof' or '3dof'")
    if source_domain == target_domain:
        raise ValueError("source_domain and target_domain must differ")

    policy_path = Path(policy_dir) / f"pushball_policy_{source_domain}.zip"
    vecnorm_path = Path(policy_dir) / "vec_normalize.pkl"
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)
    if not vecnorm_path.exists():
        raise FileNotFoundError(vecnorm_path)

    ppo = PPO.load(str(policy_path), device=device)

    def make_target_env():
        return Monitor(
            make_latent_pushball_env(
                policy_domain=source_domain,
                raw_domain=target_domain,
                mappers_dir=mappers_dir,
                max_steps=max_steps,
                render_mode=None,
                device=device,
            )()
        )

    venv = DummyVecEnv([make_target_env])
    vec_norm = VecNormalize.load(str(vecnorm_path), venv=venv)
    vec_norm.training = False
    vec_norm.norm_reward = False

    successes = 0
    steps_on_success = []
    final_dist_failure = []

    for ep in range(n_episodes):
        obs = venv.reset()[0]
        done = False
        step = 0
        last_info = {}

        while not done:
            obs_norm = vec_norm.normalize_obs(obs)
            action, _ = ppo.predict(obs_norm, deterministic=True)
            obs, _, dones, infos = venv.step(action.reshape(1, -1))
            obs = obs[0]
            done = dones[0]
            last_info = infos[0]
            step += 1

        if last_info.get("target_reached", False):
            successes += 1
            steps_on_success.append(step)
        else:
            final_dist_failure.append(last_info.get("dist_ball_target", float("nan")))

    success_rate = 100.0 * successes / max(1, n_episodes)
    print(f"Transfer {source_domain} → {target_domain}: {success_rate:.1f}% success")
    if steps_on_success:
        print(f"Avg steps on success: {np.mean(steps_on_success):.1f}")
    if final_dist_failure:
        print(f"Avg final distance on failure: {np.mean(final_dist_failure):.4f} m")

    venv.close()
    return success_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-domain", choices=["2dof", "3dof"], required=True)
    parser.add_argument("--target-domain", choices=["2dof", "3dof"], required=True)
    parser.add_argument("--mappers-dir", default="./data/LSUNN/latent_reaching")
    parser.add_argument("--policy-dir", default="./data/LSUNN/latent_pushball_2dof")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    transfer_pushball_latent(**vars(args))


if __name__ == "__main__":
    main()
