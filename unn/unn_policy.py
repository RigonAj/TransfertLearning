"""
UNN Policy — PPO operating in Cartesian latent space.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from unn.bases_unn import CartesianStateEncoder, ActionMapper


class UNNPolicy:
    """
    Complete UNN transfer policy:
      - Encodes raw observation into Cartesian latent state (fixed).
      - PPO policy (trained in latent space) outputs latent action (dx, dy).
      - ActionMapper converts (joint_angles, dx, dy) into robot joint action.
    """
    def __init__(self,
                 encoder: CartesianStateEncoder,
                 ppo_policy: PPO,
                 vec_normalize: Optional[VecNormalize],
                 action_mapper: ActionMapper,
                 arm_obs_size: int,
                 n_joints: int,
                 device: str = "cpu"):
        self.encoder = encoder
        self.ppo_policy = ppo_policy
        self.vec_normalize = vec_normalize
        self.action_mapper = action_mapper
        self.arm_obs_size = arm_obs_size
        self.n_joints = n_joints
        self.device = device

        self.action_mapper.eval()
        self.ppo_policy.policy.eval()

    @torch.no_grad()
    def predict(self, raw_obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        # 1) Encode to Cartesian latent state (6D)
        latent_state = self.encoder.encode(raw_obs, self.arm_obs_size)   # (6,)

        # 2) Normalize if VecNormalize was used during training
        if self.vec_normalize is not None:
            latent_state_norm = self.vec_normalize.normalize_obs(latent_state.reshape(1, -1))
            latent_state_norm = latent_state_norm[0]
        else:
            latent_state_norm = latent_state

        # 3) PPO inference → latent action (dx, dy) in [-1,1]
        latent_action, _ = self.ppo_policy.predict(latent_state_norm, deterministic=deterministic)

        # 4) Extract joint angles from raw observation (normalized)
        joint_angles = raw_obs[:self.n_joints]   # already normalized (θ/π)

        # 5) Map latent action to joint action using ActionMapper
        ja_t = torch.tensor(joint_angles, dtype=torch.float32, device=self.device).unsqueeze(0)
        la_t = torch.tensor(latent_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        joint_action = self.action_mapper(ja_t, la_t).cpu().numpy()[0]

        return joint_action

    def save(self, save_dir: str, name: str = "unn_policy"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.ppo_policy.save(save_path / f"{name}_ppo.zip")
        if self.vec_normalize is not None:
            self.vec_normalize.save(save_path / f"{name}_vecnorm.pkl")
        torch.save(self.action_mapper.state_dict(), save_path / f"{name}_action_mapper.pt")
        # Save metadata
        meta = {
            'arm_obs_size': self.arm_obs_size,
            'n_joints': self.n_joints,
        }
        torch.save(meta, save_path / f"{name}_meta.pt")
        print(f"  UNN policy saved → {save_path}")

    @classmethod
    def load(cls, load_dir: str, name: str = "unn_policy",
             encoder: CartesianStateEncoder = None, device: str = "cpu"):
        load_path = Path(load_dir)
        # Load metadata
        meta = torch.load(load_path / f"{name}_meta.pt", map_location=device)
        # Load PPO
        ppo = PPO.load(load_path / f"{name}_ppo.zip", device=device)
        # Load VecNormalize (if exists)
        vec_norm = None
        vn_path = load_path / f"{name}_vecnorm.pkl"
        if vn_path.exists():
            from stable_baselines3.common.vec_env import DummyVecEnv
            import gymnasium as gym
            class DummyEnv:
                def __init__(self):
                    self.observation_space = gym.spaces.Box(-10, 10, (6,), dtype=np.float32)
                    self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
            dummy = DummyVecEnv([lambda: DummyEnv()])
            vec_norm = VecNormalize.load(vn_path, venv=dummy)
            vec_norm.training = False
            vec_norm.norm_reward = False
        # Load action mapper
        action_mapper = ActionMapper(n_joints=meta['n_joints']).to(device)
        action_mapper.load_state_dict(torch.load(load_path / f"{name}_action_mapper.pt", map_location=device))
        action_mapper.eval()
        # Encoder
        if encoder is None:
            encoder = CartesianStateEncoder()
        return cls(encoder, ppo, vec_norm, action_mapper,
                   arm_obs_size=meta['arm_obs_size'],
                   n_joints=meta['n_joints'],
                   device=device)
