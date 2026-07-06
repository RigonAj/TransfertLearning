"""
Train UNN PPO policy for PushBall 3-DoF in Cartesian latent space (velocity-based).

Latent action = normalized end-effector velocity (vx, vy) in [-1, 1].
The UNNLatentEnv wrapper converts it to a joint command via analytic IK:
    v_ee         = latent_action * V_MAX_EE         (m/s)
    omega_joints = J^+ @ v_ee                       (rad/s)
    joint_action = omega_joints / omega_max          (normalized, → env)
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

from envs.env_pushball_3dof import PushBallEnv_3dof
from unn.bases_unn import CartesianStateEncoder, ik_velocity


# ---------------------------------------------------------------------------
# Hyper-parameter
# ---------------------------------------------------------------------------
# omega_max * max_reach = 2.0 * 3.0 = 6.0 m/s
V_MAX_EE = 6.0   # m/s


# ---------------------------------------------------------------------------
# Latent environment wrapper
# ---------------------------------------------------------------------------

class UNNLatentEnv(gym.Wrapper):
    """
    Wraps PushBallEnv_3dof so PPO sees a 6D Cartesian latent space and
    outputs normalized EE velocities instead of joint commands.

    Observation (6D, normalized in [-1, 1]):
        [eff_x, eff_y, ball_x, ball_y, tgt_x, tgt_y] / max_reach

    Action (2D, normalized in [-1, 1]):
        [vx, vy] / V_MAX_EE   — desired end-effector velocity direction & magnitude
    """

    def __init__(self, env, encoder: CartesianStateEncoder,
                 arm_obs_size: int, n_joints: int,
                 v_max_ee: float = V_MAX_EE):
        super().__init__(env)
        self.encoder      = encoder
        self.arm_obs_size = arm_obs_size
        self.n_joints     = n_joints
        self.v_max_ee     = v_max_ee

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def step(self, latent_action):
        raw_obs = self.env._get_obs()

        # Current joint angles in radians (raw_obs stores θ/π)
        joint_angles_rad = (
            np.asarray(raw_obs[:self.n_joints], dtype=np.float32).reshape(-1) * np.pi
        )

        # Scale normalized action → EE velocity (m/s)
        v_ee = np.asarray(latent_action, dtype=np.float32).reshape(2,) * self.v_max_ee

        # IK: EE velocity → joint velocities (rad/s)
        link_lengths  = np.array([self.env.l1, self.env.l2, self.env.l3], dtype=np.float32)
        omega_joints  = ik_velocity(joint_angles_rad, v_ee, link_lengths)

        # Normalize for the environment: joint_action = omega_joints / omega_max
        joint_action = (omega_joints / float(self.env.omega_max)).astype(np.float32)
        joint_action = np.clip(joint_action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = self.env.step(joint_action)
        return self.encoder.encode(obs, self.arm_obs_size), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.encoder.encode(obs, self.arm_obs_size), info


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    N_ENVS          = 32
    TOTAL_TIMESTEPS = 50_000_000

    encoder = CartesianStateEncoder(max_reach=3.0)

    def make_env(rank: int = 0):
        def _init():
            env = PushBallEnv_3dof(render_mode=None, max_steps=150)
            env = UNNLatentEnv(env, encoder, arm_obs_size=8, n_joints=3, v_max_ee=V_MAX_EE)
            env = Monitor(env)
            env.reset(seed=rank)
            return env
        return _init

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99
    )

    eval_env = DummyVecEnv([make_env(0)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0
    )

    class SyncEvalCallback(EvalCallback):
        def _on_step(self) -> bool:
            self.eval_env.obs_rms = self.training_env.obs_rms
            self.eval_env.ret_rms = self.training_env.ret_rms
            return super()._on_step()

    eval_callback = SyncEvalCallback(
        eval_env,
        best_model_save_path="./data/UNN/unn_pushball_3dof",
        log_path="./data/UNN/logs_3dof",
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy", train_env,
        n_steps=2048, batch_size=1024, n_epochs=5,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                           activation_fn=nn.Tanh),
        tensorboard_log="./data/UNN/logs_3dof",
        verbose=1,
    )

    print("\nTraining UNN PPO on 3-DoF Cartesian latent space (velocity-based)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)

    # ------------------------------------------------------------------
    # Save final UNNPolicy
    # ------------------------------------------------------------------
    from unn.unn_policy import UNNPolicy

    tmp_env      = PushBallEnv_3dof()
    link_lengths = [float(tmp_env.l1), float(tmp_env.l2), float(tmp_env.l3)]
    omega_max    = float(tmp_env.omega_max)
    dt           = float(tmp_env.dt)

    unn_policy = UNNPolicy(
        encoder=encoder,
        ppo_policy=model,
        vec_normalize=train_env,
        arm_obs_size=8,
        n_joints=3,
        link_lengths=link_lengths,
        omega_max=omega_max,
        v_max_ee=V_MAX_EE,   # <-- saved in metadata
        dt=dt,
        device=DEVICE,
    )
    unn_policy.save("./data/UNN/unn_pushball_3dof", name="final")

    print("Training complete.")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
