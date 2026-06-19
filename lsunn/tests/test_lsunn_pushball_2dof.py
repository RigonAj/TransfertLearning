"""
Test LS-UNN policy trained on PushBall 2-DoF (no transfer).
The policy operates in the VAE latent space.

Usage:
    python -m lsunn.tests.test_lsunn_pushball_2dof
"""

import sys
import warnings
import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings("ignore", message=".*shared CUDA tensors.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from lsunn.latent_envs import LatentPushBallEnv


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPISODES = 2000
    MAX_STEPS = 150

    MAPPERS_DIR = Path("./data/LSUNN/latent_reaching")
    UNN_MODEL_DIR = Path("./data/LSUNN/latent_pushball_2dof")
    PPO_PATH = UNN_MODEL_DIR / "pushball_policy_2dof.zip"
    VECNORM_PATH = UNN_MODEL_DIR / "vec_normalize.pkl"

    def make_latent_env():
        return Monitor(
            LatentPushBallEnv(
                policy_domain="2dof",
                raw_domain="2dof",
                mappers_dir=str(MAPPERS_DIR),
                max_steps=MAX_STEPS,
                render_mode=None,
                device=DEVICE,
            )
        )

    venv = DummyVecEnv([make_latent_env])
    vec_norm = VecNormalize.load(str(VECNORM_PATH), venv=venv)
    vec_norm.training = False
    vec_norm.norm_reward = False

    ppo_policy = PPO.load(str(PPO_PATH), device=DEVICE)

    successes = 0
    steps_on_success = []
    final_dist_failure = []

    print(f"\nRunning {NUM_EPISODES} test episodes (latent PushBall 2-DoF, no transfer)...\n")

    for ep in range(NUM_EPISODES):
        obs = vec_norm.reset()[0]
        done = False
        step_count = 0
        info_last = {}

        while not done:
            action = ppo_policy.predict(obs, deterministic=True)
            obs, _, dones, infos = vec_norm.step(action.reshape(1, -1))
            obs = obs[0]
            step_count += 1
            done = dones[0]
            info_last = infos[0]

        if info_last.get("target_reached", False):
            successes += 1
            steps_on_success.append(step_count)
        else:
            final_dist_failure.append(info_last.get("dist_ball_target", float("nan")))

        if (ep + 1) % 500 == 0:
            print(f"  {ep+1}/{NUM_EPISODES} – success rate: {100 * successes / (ep + 1):.1f}%")

    success_rate = 100 * successes / NUM_EPISODES
    print("\n" + "=" * 60)
    print("  Latent PushBall 2-DoF Test Results")
    print("=" * 60)
    print(f"  Episodes tested    : {NUM_EPISODES}")
    print(f"  Successes          : {successes}")
    print(f"  Success rate       : {success_rate:.1f}%")
    print("-" * 60)
    if steps_on_success:
        print(f"  Avg steps (success): {np.mean(steps_on_success):.1f}")
        print(f"  Steps min/max      : {np.min(steps_on_success)} / {np.max(steps_on_success)}")
    if final_dist_failure:
        print(f"  Avg dist (failure) : {np.mean(final_dist_failure):.4f} m")
        print(f"  Dist min/max       : {np.min(final_dist_failure):.4f} / {np.max(final_dist_failure):.4f} m")
    print("=" * 60)

    vec_norm.close()


if __name__ == "__main__":
    main()
