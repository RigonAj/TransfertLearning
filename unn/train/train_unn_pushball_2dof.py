"""
Train UNN PPO policy for PushBall 2-DoF in Cartesian latent space.
"""

import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from torch import nn

from envs.env_pushball_2dof import PushBallEnv_2dof
from unn.bases_unn import CartesianStateEncoder, ActionMapper

class UNNLatentEnv(gym.Wrapper):
    def __init__(self, env, encoder, action_mapper, arm_obs_size, n_joints, max_reach=3.0):
        super().__init__(env)
        self.encoder = encoder
        self.action_mapper = action_mapper
        self.arm_obs_size = arm_obs_size
        self.n_joints = n_joints
        self.action_mapper.eval()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # dx,dy

    @torch.no_grad()
    def step(self, latent_action):
        raw_obs = self.env._get_obs()
        joint_angles = raw_obs[:self.n_joints]
        # Convert latent action to joint action using mapper
        ja_t = torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0)
        la_t = torch.tensor(latent_action, dtype=torch.float32).unsqueeze(0)
        joint_action = self.action_mapper(ja_t, la_t).cpu().numpy()[0]
        # Apply to environment
        obs, reward, terminated, truncated, info = self.env.step(joint_action)
        latent_obs = self.encoder.encode(obs, self.arm_obs_size)
        return latent_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        latent_obs = self.encoder.encode(obs, self.arm_obs_size)
        return latent_obs, info

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_ENVS = 32
    TOTAL_TIMESTEPS = 50_000_000

    # Load pre‑trained action mapper for 2DoF
    mapper = ActionMapper(n_joints=2).to(DEVICE)
    mapper.load_state_dict(torch.load("./data/UNN/action_mappers/mapper_2dof.pt", map_location=DEVICE))
    mapper.eval()

    encoder = CartesianStateEncoder(max_reach=3.0)

    def make_env(rank=0):
        def _init():
            env = PushBallEnv_2dof(render_mode=None, max_steps=150)
            env = UNNLatentEnv(env, encoder, mapper, arm_obs_size=6, n_joints=2)
            env = Monitor(env)
            env.reset(seed=rank)
            return env
        return _init

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    eval_env = DummyVecEnv([make_env(0)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

    class SyncEvalCallback(EvalCallback):
        def _on_step(self) -> bool:
            self.eval_env.obs_rms = self.training_env.obs_rms
            self.eval_env.ret_rms = self.training_env.ret_rms
            return super()._on_step()

    eval_callback = SyncEvalCallback(
        eval_env, best_model_save_path="./data/UNN/unn_pushball_2dof",
        log_path="./data/UNN/logs_2dof", eval_freq=10_000, n_eval_episodes=20, deterministic=True
    )

    model = PPO(
        "MlpPolicy", train_env,
        n_steps=2048, batch_size=1024, n_epochs=5,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256]), activation_fn=nn.Tanh),
        tensorboard_log="./data/UNN/logs_2dof", verbose=1
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)

    # Save final policy (full UNNPolicy)
    from unn.unn_policy import UNNPolicy
    unn_policy = UNNPolicy(encoder, model, train_env, mapper, arm_obs_size=6, n_joints=2, device=DEVICE)
    unn_policy.save("./data/UNN/unn_pushball_2dof", name="final")

if __name__ == "__main__":
    main()
