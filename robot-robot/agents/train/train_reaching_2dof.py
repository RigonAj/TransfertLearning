import os
import numpy as np
import multiprocessing
import torch

from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

from envs.env_reaching_2dof import ReachingEnv_2dof


# ==============================
# CPU / Threads
# ==============================
torch.set_num_threads(16)

# ==============================
# Hyperparams
# ==============================
total_batch = 16384
n_envs = 64
TOTAL_TIMESTEPS = 10_000_000 

# ==============================
# Directories
# ==============================
run_id = 1
run_name = f"ppo_reach_2dof_{run_id}"
tensorboard_log_dir = f"./data/models/{run_name}/"
model_dir = f"./data/models/{run_name}/"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

# ==============================
# Schedules
# ==============================
def linear_schedule(initial_value):
    def func(progress_remaining):
        return max(initial_value * (0.1 + 0.9 * progress_remaining), 1e-5)
    return func

def clip_schedule(progress_remaining):
    return max(0.05, 0.2 * (0.25 + 0.75 * progress_remaining))

# ==============================
# Sync VecNormalize
# ==============================
def sync_envs_normalization(train_env, eval_env):
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

# ==============================
# Callback
# ==============================
class SyncedEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        sync_envs_normalization(self.training_env, self.eval_env)
        result = super()._on_step()

        if self.last_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.last_mean_reward
            vec_path = os.path.join(self.best_model_save_path, "vec_normalize.pkl")
            self.training_env.save(vec_path)

        return result

# ==============================
# ENV FACTORY (clé du multi-env)
# ==============================
def make_env(rank, seed=0):
    def _init():
        env = ReachingEnv_2dof(render_mode=None)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

if multiprocessing.current_process().name == "MainProcess":

    # ==============================
    # TRAIN ENV (PARALLEL)
    # ==============================
    train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # ==============================
    # EVAL ENV (single)
    # ==============================
    eval_env = DummyVecEnv([make_env(0)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )

    # ==============================
    # Callback
    # ==============================
    eval_callback = SyncedEvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=10_000,
        n_eval_episodes=150,
        deterministic=True,
    )

    # ==============================
    # Policy
    # ==============================
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[256, 256],
        log_std_init=-1.0,
    )

    # ==============================
    # PPO
    # ==============================

    n_steps = total_batch // n_envs
    batch_size = 512
    
    assert (n_steps * n_envs) % batch_size == 0

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=linear_schedule(3e-4),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_schedule,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl= None,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=policy_kwargs,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(os.path.join(model_dir, "ppo_reach_final"))
    train_env.save(os.path.join(model_dir, "vec_normalize.pkl"))
