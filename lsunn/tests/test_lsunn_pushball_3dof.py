"""
Test LS-UNN policy trained on PushBall 3-DoF (no transfer).
The policy operates in the VAE latent space.

Usage:
    python -m lsunn.tests.test_lsunn_pushball_3dof
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

from envs.env_pushball_3dof import PushBallEnv_3dof
from lsunn.bases_vae import BaseVAE, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM
from lsunn.unn_policy import UNNPolicy
from lsunn.train.train_lsunn_pushball_3dof import LatentPushBallEnv_3dof


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPISODES = 2000
    MAX_STEPS = 150

    BASES_DIR = Path("./data/lsunn_bases")
    BASE_3DOF_PATH = BASES_DIR / "pushball_bases_r2_3dof.pt"

    UNN_MODEL_DIR = Path("./data/LSUNN/lsunn_pushball_3dof_1")
    PPO_PATH = UNN_MODEL_DIR / "unn_ppo_pushball_3dof.zip"
    VECNORM_PATH = UNN_MODEL_DIR / "vec_normalize.pkl"

    # Load VAE base for 3-DoF
    print("Loading LS-UNN VAE base for 3-DoF...")
    base_3dof = BaseVAE(
        state_dim=12,
        latent_dim=DEFAULT_LATENT_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM
    ).to(DEVICE)
    base_3dof.load_state_dict(
        torch.load(BASE_3DOF_PATH, map_location=DEVICE, weights_only=True)
    )
    base_3dof.eval()

    # Load PPO policy
    print("Loading UNN PPO policy...")
    ppo_policy = PPO.load(PPO_PATH, device=DEVICE)

    # Load VecNormalize — dummy env must expose latent obs space (16,)
    def make_latent_dummy():
        return Monitor(LatentPushBallEnv_3dof(base_3dof, device=DEVICE, render_mode=None))
    venv = DummyVecEnv([make_latent_dummy])
    vec_norm = VecNormalize.load(VECNORM_PATH, venv=venv)
    vec_norm.training = False
    vec_norm.norm_reward = False
    venv.close()

    # Create UNN policy
    policy = UNNPolicy(
        base_vae=base_3dof,
        ppo_policy=ppo_policy,
        vec_normalize=vec_norm,
        action_mapper=None,
        device=DEVICE
    )

    # Create raw environment for rollouts
    env = DummyVecEnv([lambda: Monitor(PushBallEnv_3dof(render_mode=None, max_steps=MAX_STEPS))])

    successes = 0
    steps_on_success = []
    final_dist_failure = []

    print(f"\nRunning {NUM_EPISODES} test episodes (LS-UNN 3-DoF, no transfer)...\n")

    for ep in range(NUM_EPISODES):
        obs = env.reset()[0]
        done = False
        step_count = 0
        info_last = {}

        while not done:
            action = policy.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action.reshape(1, -1))
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
    print("  LS-UNN 3-DoF Test Results")
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

    env.close()


if __name__ == "__main__":
    main()
