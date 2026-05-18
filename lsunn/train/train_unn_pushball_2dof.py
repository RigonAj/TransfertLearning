"""
Train LS-UNN Policy for PushBall 2-DoF (source agent).

The PPO agent is trained in the shared latent space produced by the
pre-trained VAE base for the 2-DoF robot.

Usage:
    python -m lsunn.train_unn_pushball_2dof
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from torch import nn

from envs.env_pushball_2dof import PushBallEnv_2dof
from lsunn.bases_vae import BaseVAE, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM


# ============================================================================
# Latent-space environment wrapper
# ============================================================================
class LatentPushBallEnv_2dof(gym.Wrapper):
    """
    Wraps PushBallEnv_2dof to output latent states instead of raw observations.
    The VAE encoder is frozen during PPO training.
    """
    def __init__(self, base_vae: BaseVAE, render_mode=None, max_steps=150):
        env = PushBallEnv_2dof(render_mode=render_mode, max_steps=max_steps)
        super().__init__(env)
        self.base_vae = base_vae
        self.base_vae.eval()

        # Observation space is now latent space
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(base_vae.latent_dim,),
            dtype=np.float32
        )

    @torch.no_grad()
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        latent_obs = self.base_vae.encode_np(obs)
        return latent_obs, reward, terminated, truncated, info

    @torch.no_grad()
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        latent_obs = self.base_vae.encode_np(obs)
        return latent_obs, info


# ============================================================================
# Main
# ============================================================================
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_ENVS = 32
    TOTAL_TIMESTEPS = 50_000_000  # Reduced since latent space is simpler

    RUN_ID = "lsunn_pushball_2dof_1"
    MODEL_DIR = f"./data/LSUNN/{RUN_ID}"
    LOG_DIR = f"./data/LSUNN/{RUN_ID}"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Load pre-trained VAE base ---
    print("Loading VAE base for 2-DoF...")
    base_vae = BaseVAE(
        state_dim=10,  # 6 arm + 4 task
        latent_dim=DEFAULT_LATENT_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM
    ).to(DEVICE)

    bases_dir = Path("./data/lsunn_bases")
    base_vae.load_state_dict(
        torch.load(bases_dir / "pushball_bases_r1_2dof.pt", map_location=DEVICE)
    )
    base_vae.eval()

    # --- Create latent environment ---
    def make_env(rank=0):
        def _init():
            env = LatentPushBallEnv_2dof(base_vae, render_mode=None, max_steps=150)
            env = Monitor(env)
            env.reset(seed=rank)
            return env
        return _init

    # Training env
    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99
    )

    # Eval env
    eval_env = DummyVecEnv([make_env(0)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99,
        training=False
    )

    # --- Sync callback ---
    class SyncNormEvalCallback(EvalCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_mean_reward = -np.inf

        def _on_step(self) -> bool:
            self.eval_env.obs_rms = self.training_env.obs_rms
            self.eval_env.ret_rms = self.training_env.ret_rms
            result = super()._on_step()
            if self.last_mean_reward > self.best_mean_reward:
                self.best_mean_reward = self.last_mean_reward
                vec_path = os.path.join(self.best_model_save_path, "vec_normalize.pkl")
                self.training_env.save(vec_path)
            return result

    eval_callback = SyncNormEvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    # --- PPO in latent space ---
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.Tanh,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=2048,
        batch_size=1024,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    print(f"\nTraining LS-UNN PPO on 2-DoF latent space ({DEFAULT_LATENT_DIM}D)...")
    print(f"  Model dir: {MODEL_DIR}")
    print(f"  Log dir:   {LOG_DIR}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )

    # Save
    model.save(os.path.join(MODEL_DIR, "unn_ppo_pushball_2dof"))
    train_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    print("Training complete.")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
