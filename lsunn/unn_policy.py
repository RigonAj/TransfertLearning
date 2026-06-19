"""
LS-UNN Policy — PPO agent operating in the shared latent space.

The UNN policy:
  1. Encodes raw observation to latent z using the robot's VAE encoder
  2. Runs PPO inference in latent space
  3. Outputs action in the robot's original action space

Training:
  - The PPO policy sees latent states (z) instead of raw states
  - Actions are in the original action space of the training robot
  - VecNormalize operates on latent states

Transfer:
  - Target robot encodes its state with its own VAE encoder
  - Uses the same PPO policy (trained on source) in latent space
  - Optionally uses an action mapper or trains a separate action decoder
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from typing import Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from lsunn.bases_vae import BaseVAE


class LatentStateWrapper:
    """
    Wraps an environment to output latent states instead of raw observations.
    Used during PPO training so the policy learns in latent space.
    """
    def __init__(self, base_vae: BaseVAE, device: str = "cpu"):
        self.base_vae = base_vae
        self.device = device
        self.latent_dim = base_vae.latent_dim

    @torch.no_grad()
    def encode(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        z = self.base_vae.encode(t)
        return z.cpu().numpy()


class UNNPolicy:
    """
    Complete UNN transfer policy combining:
      - A robot-specific VAE base (encoder)
      - A PPO policy trained in latent space
      - An optional action mapper for cross-robot transfer
    """
    def __init__(self, base_vae: BaseVAE, ppo_policy: PPO,
                 vec_normalize: Optional[VecNormalize] = None,
                 action_mapper: Optional[nn.Module] = None,
                 device: str = "cpu"):
        self.device = device
        self.base_vae = base_vae
        self.ppo_policy = ppo_policy
        self.vec_normalize = vec_normalize
        self.action_mapper = action_mapper

        self.base_vae.eval()

    @torch.no_grad()
    def predict(self, raw_obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Full LS-UNN forward pass:
          1. Encode raw observation → latent z
          2. Normalize latent z (if VecNormalize available)
          3. PPO policy → action
          4. Optionally map action (for cross-robot transfer)
        """
        # Step 1: Encode to latent
        z = self.base_vae.encode_np(raw_obs)

        # Step 2: Normalize (for training consistency)
        if self.vec_normalize is not None:
            z_norm = self.vec_normalize.normalize_obs(z)
        else:
            z_norm = z

        # Step 3: PPO inference
        action, _ = self.ppo_policy.predict(z_norm, deterministic=deterministic)

        # Step 4: Action mapping (for transfer to different action space)
        if self.action_mapper is not None:
            arm_obs = raw_obs[:self.base_vae.state_dim] if raw_obs.ndim == 1 else raw_obs[:, :self.base_vae.state_dim]
            a_t = torch.tensor(action, dtype=torch.float32, device=self.device)
            s_t = torch.tensor(arm_obs, dtype=torch.float32, device=self.device)
            if s_t.ndim == 1:
                s_t = s_t.unsqueeze(0)
            if a_t.ndim == 1:
                a_t = a_t.unsqueeze(0)
            action = self.action_mapper(s_t, a_t).cpu().numpy()

        return action

    def save(self, save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.ppo_policy.save(save_path / "unn_ppo_policy.zip")
        if self.vec_normalize is not None:
            self.vec_normalize.save(save_path / "unn_vec_normalize.pkl")
        if self.action_mapper is not None:
            torch.save(self.action_mapper.state_dict(), save_path / "unn_action_mapper.pt")
        print(f"  UNN Policy saved → {save_path}")

    @classmethod
    def load(cls, save_dir: str, base_vae: BaseVAE,
             action_mapper: Optional[nn.Module] = None,
             device: str = "cpu"):
        save_path = Path(save_dir)
        ppo = PPO.load(save_path / "unn_ppo_policy.zip", device=device)

        vec_norm = None
        vn_path = save_path / "unn_vec_normalize.pkl"
        if vn_path.exists():
            from stable_baselines3.common.vec_env import DummyVecEnv
            from gymnasium import spaces
            import numpy as np

            # Create a dummy env for VecNormalize loading
            class DummyEnv:
                def __init__(self):
                    self.observation_space = spaces.Box(
                        low=-10, high=10, shape=(base_vae.latent_dim,), dtype=np.float32
                    )
                    self.action_space = spaces.Box(
                        low=-1, high=1, shape=(1,), dtype=np.float32
                    )
                def reset(self, **kwargs):
                    return np.zeros(base_vae.latent_dim, dtype=np.float32), {}
                def step(self, action):
                    return np.zeros(base_vae.latent_dim, dtype=np.float32), 0, False, False, {}

            dummy_venv = DummyVecEnv([lambda: DummyEnv()])
            vec_norm = VecNormalize.load(vn_path, venv=dummy_venv)
            vec_norm.training = False
            vec_norm.norm_reward = False
            dummy_venv.close()

        am = action_mapper
        if am is None:
            am_path = save_path / "unn_action_mapper.pt"
            if am_path.exists():
                # We'd need the mapper class here; skip for simplicity
                pass

        return cls(base_vae, ppo, vec_norm, am, device)


# ============================================================================
# Action Mapper for cross-robot transfer
# ============================================================================
class LatentActionMapper(nn.Module):
    """
    Maps source action space → target action space, conditioned on target state.
    Used when transferring between robots with different action dimensions.

    Input: [target_state, source_action] → target_action
    """
    def __init__(self, state_dim: int, src_action_dim: int, tgt_action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + src_action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, tgt_action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
