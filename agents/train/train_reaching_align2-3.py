import os
import numpy as np
import multiprocessing
import torch

from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from envs.env_reaching_2dof import ReachingEnv_2dof
from envs.env_reaching_3dof import ReachingEnv_3dof
from envs.reaching_alignment_3to2 import ReachingAlignment3to2Env


# ==========================================================
# SETUP CPU
# ==========================================================
torch.set_num_threads(16)

# ==========================================================
# PARAMS
# ==========================================================
n_envs = 32
TOTAL_TIMESTEPS = 3_000_000

run_id = 1
run_name = f"align_3to2_{run_id}"

model_dir = f"./data/models/{run_name}/"
log_dir = f"./data/models/{run_name}/logs/"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


# ==========================================================
# SCHEDULES
# ==========================================================
def lr_schedule(progress):
    return 3e-4 * (0.2 + 0.8 * progress)


def clip_schedule(progress):
    return max(0.05, 0.2 * progress)


# ==========================================================
# ENV FACTORY
# ==========================================================
def make_env(rank=0):

    def _init():

        # ==========================
        # student env (2DoF)
        # ==========================
        env2 = ReachingEnv_2dof()

        # ==========================
        # expert env (3DoF)
        # ==========================
        env3 = ReachingEnv_3dof()

        # PPO expert 3DoF (gelé)
        expert_model = PPO.load(
            "./data/models/ppo_reach_3dof_1/ppo_reach_3dof_final.zip"
        )

        env = ReachingAlignment3to2Env(
            expert_env=env3,
            expert_policy=expert_model,
            lambda_pos=0.5,
            lambda_vel=0.1,
        )

        return Monitor(env)

    return _init


# ==========================================================
# MAIN
# ==========================================================
if multiprocessing.current_process().name == "MainProcess":

    # ==========================
    # TRAIN ENV
    # ==========================
    train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # ==========================
    # EVAL ENV
    # ==========================
    eval_env = DummyVecEnv([make_env(0)])

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )

    eval_env.obs_rms = train_env.obs_rms

    # ==========================
    # CALLBACK
    # ==========================
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=10_000,
        n_eval_episodes=50,
        deterministic=True,
    )

    # ==========================
    # POLICY
    # ==========================
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[256, 256, 256],
        log_std_init=-1.0,
    )

    # ==========================
    # LOAD 2DOF STUDENT
    # ==========================
    model = PPO.load(
        "./data/models/ppo_reach_2dof_1/ppo_reach_final.zip",
        env=train_env,
        device="auto",
    )

    # fine-tuning stable
    model.learning_rate = lr_schedule
    model.clip_range = clip_schedule
    model.ent_coef = 0.0005

    # ==========================
    # TRAINING
    # ==========================
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )

    # ==========================
    # SAVE
    # ==========================
    model.save(os.path.join(model_dir, "align_3to2_final"))
    train_env.save(os.path.join(model_dir, "vecnormalize.pkl"))

    print("Alignment 3to2 finished")