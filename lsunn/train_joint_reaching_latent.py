"""
Phase 1 : Entraînement joint reaching + espace latent + mappers.

Ce script apprend en parallèle :
  1. deux politiques PPO de reaching (2DoF et 3DoF) dans un espace latent partagé ;
  2. deux VAE (une par robot) qui projettent les observations de reaching vers ce latent ;
  3. deux state-mappers 2↔3 ;
  4. deux action-mappers 2↔3.

À la fin, les mappers et les VAE sont sauvegardés comme espace latent commun.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.nn import functional as F
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_reaching_2dof import ReachingEnv_2dof
from envs.env_reaching_3dof import ReachingEnv_3dof
from envs.env_continuous_reaching_2dof import Arm2DoFPersistentEnv
from envs.env_continuous_reaching_3dof import Arm3DoFPersistentEnv
from lsunn.bases_vae import BaseVAE
from lsunn.mapper_models import (
    ACTION_DIM_2DOF,
    ACTION_DIM_3DOF,
    ARM_OBS_2DOF,
    ARM_OBS_3DOF,
    ActionMapperMLP,
    MAX_REACH,
    StateMapperMLP,
    save_latent_mappers,
)


# ── Dimensions ───────────────────────────────────────────────────────────────
ARM_OBS_2DOF = 6
ARM_OBS_3DOF = 8
REACHING_OBS_2DOF = 11
REACHING_OBS_3DOF = 13
ACTION_DIM_2DOF = 2
ACTION_DIM_3DOF = 3

MAX_REACH = 3.0
OMEGA_MAX = 2.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return max(initial_value * (0.1 + 0.9 * progress_remaining), 1e-5)
    return func


def clip_schedule(progress_remaining: float) -> float:
    return max(0.05, 0.2 * (0.25 + 0.75 * progress_remaining))


# ── Latent env wrapper ────────────────────────────────────────────────────────

class LatentReachingEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, base_vae: BaseVAE, device: str = "cpu", latent_dim: int = 16):
        super().__init__(env)
        self.base_vae = base_vae
        self.device = device
        self.latent_dim = latent_dim
        self.base_vae.eval()
        self.observation_space = spaces.Box(-10.0, 10.0, (latent_dim,), dtype=np.float32)
        self.last_raw_obs: Optional[np.ndarray] = None
        self.last_raw_action: Optional[np.ndarray] = None

    @torch.no_grad()
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_raw_obs = obs.copy()
        self.last_raw_action = None
        return self._encode(obs), info

    @torch.no_grad()
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_raw_obs = obs.copy()
        self.last_raw_action = np.asarray(action, dtype=np.float32).copy()
        return self._encode(obs), reward, terminated, truncated, info

    def get_raw_obs(self) -> Optional[np.ndarray]:
        return self.last_raw_obs.copy() if self.last_raw_obs is not None else None

    def get_raw_action(self) -> Optional[np.ndarray]:
        return self.last_raw_action.copy() if self.last_raw_action is not None else None

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.base_vae.encode(obs_t).detach().cpu().numpy().flatten().astype(np.float32)


# ── Pseudo-labels / IK helpers ────────────────────────────────────────────────

def _ik_2dof(x: torch.Tensor, y: torch.Tensor, l1: float, l2: float) -> Tuple[torch.Tensor, torch.Tensor]:
    r = torch.sqrt(x.pow(2) + y.pow(2)).clamp(max=l1 + l2 - 1e-6)
    cos_theta2 = (r.pow(2) - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = cos_theta2.clamp(-1.0, 1.0)
    theta2 = torch.atan2(torch.sqrt(1 - cos_theta2.pow(2)).clamp_min(0.0), cos_theta2)
    theta1 = torch.atan2(y, x) - torch.atan2(l2 * torch.sin(theta2), l1 + l2 * torch.cos(theta2))
    return theta1, theta2


def _pseudo_2to3_state(arm2: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        eff_x = arm2[:, 4] * MAX_REACH
        eff_y = arm2[:, 5] * MAX_REACH
        theta1, theta2 = _ik_2dof(eff_x, eff_y, l1=1.0, l2=2.0)
        dtheta1 = arm2[:, 2] * OMEGA_MAX
        dtheta2 = arm2[:, 3] * OMEGA_MAX
        return torch.stack(
            [
                theta1 / np.pi,
                theta2 / np.pi,
                torch.zeros_like(theta1),
                dtheta1 / OMEGA_MAX,
                dtheta2 / OMEGA_MAX,
                torch.zeros_like(theta1),
                arm2[:, 4],
                arm2[:, 5],
            ],
            dim=1,
        )


def _pseudo_3to2_state(arm3: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        eff_x = arm3[:, 6] * MAX_REACH
        eff_y = arm3[:, 7] * MAX_REACH
        theta1, theta2 = _ik_2dof(eff_x, eff_y, l1=1.5, l2=1.5)
        dtheta1 = arm3[:, 3] * OMEGA_MAX
        dtheta2 = arm3[:, 4] * OMEGA_MAX
        return torch.stack(
            [
                theta1 / np.pi,
                theta2 / np.pi,
                dtheta1 / OMEGA_MAX,
                dtheta2 / OMEGA_MAX,
                arm3[:, 6],
                arm3[:, 7],
            ],
            dim=1,
        )


def _pseudo_2to3_action(action2: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([action2, torch.zeros_like(action2[:, :1])], dim=1)


def _pseudo_3to2_action(action3: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return action3[:, :2]


# ── Trainer ───────────────────────────────────────────────────────────────────

class JointReachingLatentTrainer:
    def __init__(
        self,
        run_id: str,
        save_dir: Path,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        n_envs: int = 32,
        n_steps: int = 256,
        batch_size: int = 512,
        n_epochs: int = 5,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.001,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        vae_alpha: float = 10.0,
        vae_beta: float = 0.001,
        vae_gamma: float = 1.0,
        vae_lambda: float = 1.0,
        state_mapper_weight: float = 0.5,
        action_mapper_weight: float = 0.5,
        pseudo_weight: float = 0.1,
        weight_decay: float = 1e-5,
        persistent: bool = False,
        device: str = "cpu",
        seed: int = 0,
        save_freq: int = 10,
    ):
        self.run_id = run_id
        self.save_dir = Path(save_dir) / run_id
        self.ckpt_dir = self.save_dir / "checkpoints"
        self.tb_dir = self.save_dir / "tb"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.vae_alpha = vae_alpha
        self.vae_beta = vae_beta
        self.vae_gamma = vae_gamma
        self.vae_lambda = vae_lambda
        self.state_mapper_weight = state_mapper_weight
        self.action_mapper_weight = action_mapper_weight
        self.pseudo_weight = pseudo_weight
        self.weight_decay = weight_decay
        self.persistent = persistent
        self.seed = seed
        self.save_freq = save_freq

        self.base_2dof = BaseVAE(REACHING_OBS_2DOF, latent_dim, hidden_dim).to(self.device)
        self.base_3dof = BaseVAE(REACHING_OBS_3DOF, latent_dim, hidden_dim).to(self.device)

        self.state_mapper_2to3 = StateMapperMLP(ARM_OBS_2DOF, ARM_OBS_3DOF, hidden_dim).to(self.device)
        self.state_mapper_3to2 = StateMapperMLP(ARM_OBS_3DOF, ARM_OBS_2DOF, hidden_dim).to(self.device)
        self.action_mapper_2to3 = ActionMapperMLP(ARM_OBS_3DOF, ACTION_DIM_2DOF, ACTION_DIM_3DOF, hidden_dim).to(self.device)
        self.action_mapper_3to2 = ActionMapperMLP(ARM_OBS_2DOF, ACTION_DIM_3DOF, ACTION_DIM_2DOF, hidden_dim).to(self.device)

        self.optimizer = optim.Adam(
            list(self.base_2dof.parameters())
            + list(self.base_3dof.parameters())
            + list(self.state_mapper_2to3.parameters())
            + list(self.state_mapper_3to2.parameters())
            + list(self.action_mapper_2to3.parameters())
            + list(self.action_mapper_3to2.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.env_cls_2dof = Arm2DoFPersistentEnv if persistent else ReachingEnv_2dof
        self.env_cls_3dof = Arm3DoFPersistentEnv if persistent else ReachingEnv_3dof

        self.env_2dof = self._build_env(self.env_cls_2dof, self.base_2dof, seed)
        self.env_3dof = self._build_env(self.env_cls_3dof, self.base_3dof, seed + 1000)

        self.ppo_2dof = self._build_ppo(self.env_2dof, ACTION_DIM_2DOF)
        self.ppo_3dof = self._build_ppo(self.env_3dof, ACTION_DIM_3DOF)

        self._save_config(0)

    def _build_env(self, env_cls, base_vae: BaseVAE, seed: int):
        def make_env(rank: int):
            def _init():
                env = env_cls(render_mode=None)
                env = LatentReachingEnv(env, base_vae, str(self.device), self.latent_dim)
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
                return env
            return _init

        env = SubprocVecEnv([make_env(i) for i in range(self.n_envs)])
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=self.gamma,
        )
        return env

    def _build_ppo(self, env, action_dim: int):
        policy_kwargs = dict(
            activation_fn=nn.Tanh,
            net_arch=[256, 256, 256],
            log_std_init=-1.0,
        )
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(self.learning_rate),
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=clip_schedule,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            target_kl=None,
            verbose=1,
            tensorboard_log=str(self.tb_dir),
            policy_kwargs=policy_kwargs,
            device=self.device,
        )

    def _predict_action(self, ppo: PPO, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=ppo.device)
            try:
                actions, values, log_probs = ppo.policy(obs_t, deterministic=deterministic)
            except Exception:
                actions = ppo.policy(obs_t, deterministic=deterministic)
                values = ppo.policy.get_value(obs_t)
                dist = ppo.policy.get_distribution(obs_t)
                log_probs = dist.log_prob(actions)
        return (
            actions.cpu().numpy().astype(np.float32),
            values.cpu().numpy().flatten().astype(np.float32),
            log_probs.cpu().numpy().flatten().astype(np.float32),
        )

    def _predict_value(self, ppo: PPO, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=ppo.device)
            try:
                _, values, _ = ppo.policy(obs_t, deterministic=True)
            except Exception:
                values = ppo.policy.get_value(obs_t)
        return values.cpu().numpy().flatten().astype(np.float32)

    def _collect_rollout(self, ppo: PPO, vec_env, n_steps: int) -> Dict[str, object]:
        n_envs = vec_env.num_envs
        obs = vec_env.reset()
        raw_obs = np.stack(vec_env.venv.get_attr("get_raw_obs"))

        episode_starts = np.ones((n_envs,), dtype=np.float32)
        observations: List[np.ndarray] = []
        raw_before: List[np.ndarray] = []
        raw_after: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        rewards: List[np.ndarray] = []
        dones: List[np.ndarray] = []
        values: List[np.ndarray] = []
        log_probs: List[np.ndarray] = []
        infos: List[List[dict]] = []
        episode_starts_list: List[np.ndarray] = []

        for _ in range(n_steps):
            episode_starts_list.append(episode_starts.copy())
            actions_np, values_np, log_probs_np = self._predict_action(ppo, obs, deterministic=False)
            observations.append(obs)
            raw_before.append(raw_obs)
            actions.append(actions_np)
            values.append(values_np)
            log_probs.append(log_probs_np)

            obs, rewards_np, dones_np, infos_np = vec_env.step(actions_np)
            raw_obs = np.stack(vec_env.venv.get_attr("get_raw_obs"))

            raw_after.append(raw_obs)
            rewards.append(rewards_np)
            dones.append(dones_np)
            infos.append(infos_np)
            episode_starts = dones_np.astype(np.float32)

        last_value = self._predict_value(ppo, obs)
        last_episode_done = np.asarray(dones[-1], dtype=np.float32)

        buffer = RolloutBuffer(
            n_steps,
            vec_env.observation_space,
            vec_env.action_space,
            device=ppo.device,
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            n_envs=n_envs,
        )
        buffer.reset()
        for i in range(n_steps):
            buffer.add(
                observations[i],
                actions[i],
                rewards[i],
                episode_starts_list[i],
                values[i],
                log_probs[i],
            )
        buffer.compute_returns_and_advantage(last_value, last_episode_done)
        ppo.rollout_buffer = buffer
        ppo.train()

        success_flags = []
        dists = []
        for step_infos in infos:
            for info in step_infos:
                success_flags.append(bool(info.get("target_reached", False)))
                if "dist" in info:
                    dists.append(float(info["dist"]))

        return {
            "raw_before": np.stack(raw_before, axis=0),
            "raw_after": np.stack(raw_after, axis=0),
            "actions": np.stack(actions, axis=0),
            "rewards": np.stack(rewards, axis=0),
            "dones": np.stack(dones, axis=0),
            "infos": infos,
            "mean_reward": float(np.mean(rewards)),
            "success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
            "mean_dist": float(np.mean(dists)) if dists else float("nan"),
        }

    def _all_parameters(self):
        return (
            list(self.base_2dof.parameters())
            + list(self.base_3dof.parameters())
            + list(self.state_mapper_2to3.parameters())
            + list(self.state_mapper_3to2.parameters())
            + list(self.action_mapper_2to3.parameters())
            + list(self.action_mapper_3to2.parameters())
        )

    def _update_latent_models(self, rollout_2: Dict[str, object], rollout_3: Dict[str, object]) -> Dict[str, float]:
        raw2_before = np.asarray(rollout_2["raw_before"], dtype=np.float32).reshape(-1, REACHING_OBS_2DOF)
        raw2_after = np.asarray(rollout_2["raw_after"], dtype=np.float32).reshape(-1, REACHING_OBS_2DOF)
        raw3_before = np.asarray(rollout_3["raw_before"], dtype=np.float32).reshape(-1, REACHING_OBS_3DOF)
        raw3_after = np.asarray(rollout_3["raw_after"], dtype=np.float32).reshape(-1, REACHING_OBS_3DOF)

        raw2 = np.concatenate([raw2_before, raw2_after], axis=0)
        raw3 = np.concatenate([raw3_before, raw3_after], axis=0)

        arm2 = raw2_before[:, :ARM_OBS_2DOF]
        arm3 = raw3_before[:, :ARM_OBS_3DOF]
        act2 = np.asarray(rollout_2["actions"], dtype=np.float32).reshape(-1, ACTION_DIM_2DOF)
        act3 = np.asarray(rollout_3["actions"], dtype=np.float32).reshape(-1, ACTION_DIM_3DOF)

        raw2_t = torch.as_tensor(raw2, dtype=torch.float32, device=self.device)
        raw3_t = torch.as_tensor(raw3, dtype=torch.float32, device=self.device)
        arm2_t = torch.as_tensor(arm2, dtype=torch.float32, device=self.device)
        arm3_t = torch.as_tensor(arm3, dtype=torch.float32, device=self.device)
        act2_t = torch.as_tensor(act2, dtype=torch.float32, device=self.device)
        act3_t = torch.as_tensor(act3, dtype=torch.float32, device=self.device)

        self.base_2dof.train()
        self.base_3dof.train()
        self.state_mapper_2to3.train()
        self.state_mapper_3to2.train()
        self.action_mapper_2to3.train()
        self.action_mapper_3to2.train()

        z2, mu2, logvar2 = self.base_2dof.encoder(raw2_t)
        z3, mu3, logvar3 = self.base_3dof.encoder(raw3_t)

        recon2 = self.base_2dof.decoder(z2)
        recon3 = self.base_3dof.decoder(z3)
        cross2 = self.base_2dof.decoder(z3)
        cross3 = self.base_3dof.decoder(z2)

        loss_recon = F.mse_loss(recon2, raw2_t) + F.mse_loss(recon3, raw3_t)
        loss_cross = F.mse_loss(cross2, raw2_t) + F.mse_loss(cross3, raw3_t)
        loss_kl = (
            -0.5 * torch.mean(torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), dim=1))
            -0.5 * torch.mean(torch.sum(1 + logvar3 - mu3.pow(2) - logvar3.exp(), dim=1))
        )
        loss_sim = F.mse_loss(z2, z3)
        loss_vae = self.vae_alpha * loss_recon + self.vae_beta * loss_kl + self.vae_gamma * loss_sim + self.vae_lambda * loss_cross

        s2_hat = self.state_mapper_2to3(arm2_t)
        s2_recon = self.state_mapper_3to2(s2_hat)
        s3_hat = self.state_mapper_3to2(arm3_t)
        s3_recon = self.state_mapper_2to3(s3_hat)
        loss_state_cycle = F.mse_loss(s2_recon, arm2_t) + F.mse_loss(s3_recon, arm3_t)

        s2_pseudo = _pseudo_2to3_state(arm2_t)
        s3_pseudo = _pseudo_3to2_state(arm3_t)
        loss_state_sup = F.mse_loss(s2_hat, s2_pseudo) + F.mse_loss(s3_hat, s3_pseudo)
        loss_state = loss_state_cycle + self.pseudo_weight * loss_state_sup

        a3_from_2 = self.action_mapper_2to3(arm3_t, act2_t)
        a2_recon = self.action_mapper_3to2(arm2_t, a3_from_2)
        a2_from_3 = self.action_mapper_3to2(arm2_t, act3_t)
        a3_recon = self.action_mapper_2to3(arm3_t, a2_from_3)
        loss_action_cycle = F.mse_loss(a2_recon, act2_t) + F.mse_loss(a3_recon, act3_t)

        a3_pseudo = _pseudo_2to3_action(act2_t)
        a2_pseudo = _pseudo_3to2_action(act3_t)
        loss_action_sup = F.mse_loss(a3_from_2, a3_pseudo) + F.mse_loss(a2_from_3, a2_pseudo)
        loss_action = loss_action_cycle + self.pseudo_weight * loss_action_sup

        loss_total = loss_vae + self.state_mapper_weight * loss_state + self.action_mapper_weight * loss_action

        self.optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        nn.utils.clip_grad_norm_(self._all_parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "vae": float(loss_vae.item()),
            "state": float(loss_state.item()),
            "action": float(loss_action.item()),
            "total": float(loss_total.item()),
            "sim": float(loss_sim.item()),
        }

    def _save_config(self, label: object) -> None:
        config = {
            "run_id": self.run_id,
            "label": str(label),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "n_envs": self.n_envs,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "vae_alpha": self.vae_alpha,
            "vae_beta": self.vae_beta,
            "vae_gamma": self.vae_gamma,
            "vae_lambda": self.vae_lambda,
            "state_mapper_weight": self.state_mapper_weight,
            "action_mapper_weight": self.action_mapper_weight,
            "pseudo_weight": self.pseudo_weight,
            "weight_decay": self.weight_decay,
            "persistent": self.persistent,
            "device": str(self.device),
            "seed": self.seed,
            "save_freq": self.save_freq,
        }
        with open(self.save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def save_checkpoint(self, label: object = "latest") -> None:
        ckpt_dir = self.ckpt_dir / str(label)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.base_2dof.state_dict(), ckpt_dir / "base_2dof.pt")
        torch.save(self.base_3dof.state_dict(), ckpt_dir / "base_3dof.pt")
        save_latent_mappers(
            str(ckpt_dir),
            self.state_mapper_2to3,
            self.state_mapper_3to2,
            self.action_mapper_2to3,
            self.action_mapper_3to2,
            run_id="latent_reaching",
        )
        save_latent_mappers(
            str(self.save_dir),
            self.state_mapper_2to3,
            self.state_mapper_3to2,
            self.action_mapper_2to3,
            self.action_mapper_3to2,
            run_id="latent_reaching",
        )
        torch.save(self.base_2dof.state_dict(), self.save_dir / "base_2dof.pt")
        torch.save(self.base_3dof.state_dict(), self.save_dir / "base_3dof.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        self.ppo_2dof.save(ckpt_dir / "ppo_2dof")
        self.ppo_3dof.save(ckpt_dir / "ppo_3dof")
        self.env_2dof.save(ckpt_dir / "vec_normalize_2dof.pkl")
        self.env_3dof.save(ckpt_dir / "vec_normalize_3dof.pkl")

        self._save_config(label)
        with open(ckpt_dir / "latest.txt", "w", encoding="utf-8") as f:
            f.write(str(label))
        print(f"Checkpoint saved → {ckpt_dir}")

    def train(self, total_timesteps: int) -> None:
        iterations = int(np.ceil(total_timesteps / (self.n_steps * self.n_envs)))
        pbar = tqdm(range(iterations), desc="Joint reaching + latent")

        for iteration in pbar:
            rollout_2 = self._collect_rollout(self.ppo_2dof, self.env_2dof, self.n_steps)
            rollout_3 = self._collect_rollout(self.ppo_3dof, self.env_3dof, self.n_steps)

            latent_metrics = self._update_latent_models(rollout_2, rollout_3)

            postfix = {
                "r2": f"{rollout_2['mean_reward']:.3f}",
                "s2": f"{rollout_2['success_rate']:.2f}",
                "d2": f"{rollout_2['mean_dist']:.3f}",
                "r3": f"{rollout_3['mean_reward']:.3f}",
                "s3": f"{rollout_3['success_rate']:.2f}",
                "d3": f"{rollout_3['mean_dist']:.3f}",
                "vae": f"{latent_metrics['vae']:.3f}",
                "state": f"{latent_metrics['state']:.3f}",
                "action": f"{latent_metrics['action']:.3f}",
                "total": f"{latent_metrics['total']:.3f}",
            }
            pbar.set_postfix(postfix)

            if (iteration + 1) % self.save_freq == 0:
                self.save_checkpoint(iteration + 1)

        self.save_checkpoint("final")

    def close(self) -> None:
        for env in (self.env_2dof, self.env_3dof):
            if env is not None:
                env.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Joint reaching + latent + mappers training")
    parser.add_argument("--run-id", type=str, default="joint_reaching_latent")
    parser.add_argument("--save-dir", type=Path, default=Path("./data/LSUNN/joint_reaching_latent"))
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--vae-alpha", type=float, default=10.0)
    parser.add_argument("--vae-beta", type=float, default=0.001)
    parser.add_argument("--vae-gamma", type=float, default=1.0)
    parser.add_argument("--vae-lambda", type=float, default=1.0)
    parser.add_argument("--state-mapper-weight", type=float, default=0.5)
    parser.add_argument("--action-mapper-weight", type=float, default=0.5)
    parser.add_argument("--pseudo-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    torch.set_num_threads(args.torch_threads)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    seed_everything(args.seed)

    trainer = JointReachingLatentTrainer(
        run_id=args.run_id,
        save_dir=args.save_dir,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        vae_alpha=args.vae_alpha,
        vae_beta=args.vae_beta,
        vae_gamma=args.vae_gamma,
        vae_lambda=args.vae_lambda,
        state_mapper_weight=args.state_mapper_weight,
        action_mapper_weight=args.action_mapper_weight,
        pseudo_weight=args.pseudo_weight,
        weight_decay=args.weight_decay,
        persistent=args.persistent,
        device=device,
        seed=args.seed,
        save_freq=args.save_freq,
    )

    try:
        trainer.train(total_timesteps=args.total_timesteps)
    finally:
        trainer.close()

    print(f"Training finished. Models saved in {trainer.save_dir}")


if __name__ == "__main__":
    main()
