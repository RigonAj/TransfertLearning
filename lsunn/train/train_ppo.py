# lsunn/train/train_ppo.py
"""
Phase 3 : Entraînement des politiques PPO dans l'espace latent partagé.
Charge les poids VAE depuis DATA_DIR/{base_2dof,base_3dof}.pt.
Sauvegarde les politiques dans DATA_DIR/{lsunn_2dof,lsunn_3dof}_latent/.
"""

import sys
import numpy as np
from pathlib import Path

import torch
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof
from lsunn.bases_vae import BaseVAE

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("./data/LSUNN/joint_model")
LATENT_DIM = 16
HIDDEN_DIM = 128


# ── Wrapper espace latent ─────────────────────────────────────────────────────

class LatentEnv(gym.Wrapper):
    """
    Wrapper Gymnasium qui remplace l'observation brute par l'encodage
    latent issu d'un BaseVAE.
    """

    def __init__(self, env: gym.Env, base_vae: BaseVAE,
                 device: str = "cpu", latent_dim: int = 16):
        super().__init__(env)
        self.base_vae  = base_vae
        self.device    = device
        self.base_vae.eval()
        self.observation_space = spaces.Box(
            -10, 10, (latent_dim,), dtype=np.float32
        )

    @torch.no_grad()
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        latent = self._encode(obs)
        return latent, reward, terminated, truncated, info

    @torch.no_grad()
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._encode(obs), info

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        return self.base_vae.encode(obs_t).cpu().numpy().flatten()


# ── Entraînement PPO ──────────────────────────────────────────────────────────

def _train_single_policy(
    base_vae: BaseVAE,
    EnvClass,
    timesteps: int,
    run_id: str,
    device: str = "cpu",
    n_envs: int = 8,
    latent_dim: int = 16,
) -> tuple[PPO, VecNormalize]:
    """
    Entraîne une politique PPO sur l'environnement latent d'``EnvClass``.
    Sauvegarde la politique et le VecNormalize dans DATA_DIR/<run_id>/.
    """

    def make_env():
        raw = EnvClass(render_mode=None, max_steps=150)
        return LatentEnv(raw, base_vae, device, latent_dim)

    train_env = SubprocVecEnv(
        [lambda: Monitor(make_env()) for _ in range(n_envs)]
    )
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
    )

    model_dir = DATA_DIR / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    ppo = PPO(
        "MlpPolicy",
        train_env,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[128, 128]),
        device=device,
    )
    ppo.learn(total_timesteps=timesteps, progress_bar=True)

    ppo.save(model_dir / "policy")
    train_env.save(model_dir / "vec_normalize.pkl")

    return ppo, train_env


def train_latent_policies(
    base_2dof: BaseVAE,
    base_3dof: BaseVAE,
    device: str = "cpu",
    total_timesteps: int = 5_000_000,
    n_envs: int = 8,
    latent_dim: int = 16,
) -> tuple[PPO, VecNormalize, PPO, VecNormalize]:
    """
    Entraîne les politiques PPO pour 2DoF et 3DoF dans l'espace latent.

    Returns
    -------
    (ppo_2dof, vec_norm_2dof, ppo_3dof, vec_norm_3dof)
    """
    print("\n" + "=" * 60)
    print("Phase 3: Training PPO policies in shared latent space")
    print("=" * 60)

    print("\n  Training 2DoF latent policy…")
    ppo_2dof, vec_norm_2dof = _train_single_policy(
        base_2dof, PushBallEnv_2dof,
        total_timesteps // 2, "lsunn_2dof_latent",
        device, n_envs, latent_dim,
    )

    print("\n  Training 3DoF latent policy…")
    ppo_3dof, vec_norm_3dof = _train_single_policy(
        base_3dof, PushBallEnv_3dof,
        total_timesteps // 2, "lsunn_3dof_latent",
        device, n_envs, latent_dim,
    )

    return ppo_2dof, vec_norm_2dof, ppo_3dof, vec_norm_3dof


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DEVICE = "cpu"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Charger les poids VAE ─────────────────────────────────────────────
    print(f"  Loading VAE weights from {DATA_DIR} …")
    from envs.env_pushball_2dof import PushBallEnv_2dof as _E2
    from envs.env_pushball_3dof import PushBallEnv_3dof as _E3

    obs_dim_2dof = _E2(render_mode=None).observation_space.shape[0]
    obs_dim_3dof = _E3(render_mode=None).observation_space.shape[0]

    base_2dof = BaseVAE(obs_dim_2dof, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
    base_3dof = BaseVAE(obs_dim_3dof, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
    base_2dof.load_state_dict(
        torch.load(DATA_DIR / "base_2dof.pt", map_location=DEVICE)
    )
    base_3dof.load_state_dict(
        torch.load(DATA_DIR / "base_3dof.pt", map_location=DEVICE)
    )
    base_2dof.eval()
    base_3dof.eval()

    # ── Entraînement ──────────────────────────────────────────────────────
    train_latent_policies(
        base_2dof, base_3dof,
        device=DEVICE,
        total_timesteps=5_000_000,
        n_envs=8,
        latent_dim=LATENT_DIM,
    )

    print(f"\n  PPO policies saved → {DATA_DIR}/lsunn_{{2,3}}dof_latent/")


if __name__ == "__main__":
    main()
