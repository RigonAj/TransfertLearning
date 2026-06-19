from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from lsunn.mapper_models import (
    ARM_OBS_2DOF,
    ARM_OBS_3DOF,
    LATENT_PUSHBALL_DIM,
    load_latent_mappers,
)


class LatentPushBallEnv(gym.Wrapper):
    def __init__(
        self,
        policy_domain: str,
        raw_domain: str,
        mappers_dir: str,
        max_steps: int = 150,
        render_mode: Optional[str] = None,
        device: str = "cpu",
    ):
        if policy_domain not in {"2dof", "3dof"}:
            raise ValueError("policy_domain must be '2dof' or '3dof'")
        if raw_domain not in {"2dof", "3dof"}:
            raise ValueError("raw_domain must be '2dof' or '3dof'")

        if raw_domain == "2dof":
            from envs.env_pushball_2dof import PushBallEnv_2dof
            raw_env = PushBallEnv_2dof(render_mode=render_mode, max_steps=max_steps)
        else:
            from envs.env_pushball_3dof import PushBallEnv_3dof
            raw_env = PushBallEnv_3dof(render_mode=render_mode, max_steps=max_steps)

        super().__init__(raw_env)
        self.policy_domain = policy_domain
        self.raw_domain = raw_domain
        self.device = device
        self.state_2to3, self.state_3to2, _, _ = load_latent_mappers(mappers_dir, device=device)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(LATENT_PUSHBALL_DIM,),
            dtype=np.float32,
        )

    @torch.no_grad()
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._latent_from_raw(obs), info

    @torch.no_grad()
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        latent = self._latent_from_raw(obs)
        info["raw_obs"] = obs
        return latent, reward, terminated, truncated, info

    def _latent_from_raw(self, obs: np.ndarray) -> np.ndarray:
        arm_obs = obs[: self.env.arm_obs_size].astype(np.float32)
        task_obs = obs[self.env.arm_obs_size :].astype(np.float32)

        if self.raw_domain == self.policy_domain:
            policy_arm = arm_obs
            other_arm = self._map_arm(arm_obs, self.raw_domain)
        else:
            policy_arm = self._map_arm(arm_obs, self.raw_domain)
            other_arm = arm_obs

        latent = np.concatenate([policy_arm, other_arm, task_obs]).astype(np.float32)
        if latent.shape[0] != LATENT_PUSHBALL_DIM:
            raise RuntimeError(f"Expected latent dim {LATENT_PUSHBALL_DIM}, got {latent.shape[0]}")
        return np.clip(latent, -10.0, 10.0)

    def _map_arm(self, arm_obs: np.ndarray, from_domain: str) -> np.ndarray:
        arm_obs_t = torch.tensor(arm_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if from_domain == "2dof":
            mapped = self.state_2to3(arm_obs_t)
        else:
            mapped = self.state_3to2(arm_obs_t)
        return mapped.squeeze(0).detach().cpu().numpy().astype(np.float32)


def make_latent_pushball_env(
    policy_domain: str,
    raw_domain: str,
    mappers_dir: str,
    max_steps: int = 150,
    render_mode: Optional[str] = None,
    device: str = "cpu",
):
    def _init():
        return LatentPushBallEnv(
            policy_domain=policy_domain,
            raw_domain=raw_domain,
            mappers_dir=mappers_dir,
            max_steps=max_steps,
            render_mode=render_mode,
            device=device,
        )

    return _init
