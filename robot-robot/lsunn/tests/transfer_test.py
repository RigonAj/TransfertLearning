"""
Tests de transfert bidirectionnel 2DoF ↔ 3DoF avec architecture Bo_target ∘ U ∘ Bi_target.
"""

import torch
import numpy as np

from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof
from lsunn.bases_vae import BaseVAE  # ← AJOUTER CET IMPORT


@torch.no_grad()
def _test_transfer_direction(
    source_unn_policy,  # UNNPolicy avec PPO source
    target_base: BaseVAE,  # VAE cible pour Bi_target et Bo_target
    target_env_class,
    device: str,
    n_episodes: int,
    direction: str,
) -> float:
    """
    Teste une direction de transfert avec l'architecture Bo_target ∘ U ∘ Bi_target.
    
    Protocole:
        1. Bi_target(obs) → z (via le VAE cible)
        2. UNN(z, oτ) → zd (via la politique source)
        3. Bo_target(zd) → x_rd (via le VAE cible)
        4. Extraire vitesse de x_rd → action
    """
    print(f"\n  Testing transfer {direction}...")
    
    env = target_env_class(render_mode=None, max_steps=150)
    successes = 0
    target_base.eval()
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # 1. Encoder avec VAE cible → z (Bi_target)
            arm_obs = obs[:target_base.state_dim]
            z = target_base.encode_np(arm_obs, device=device, stochastic=False)
            
            # 2. Construire [z, oτ] pour l'entrée PPO
            task_obs = obs[target_base.state_dim:]
            ppo_input = np.concatenate([z.flatten(), task_obs])
            
            # 3. Normaliser (si source_unn_policy.vec_normalize existe)
            if source_unn_policy.vec_normalize is not None:
                ppo_input_norm = source_unn_policy.vec_normalize.normalize_obs(ppo_input)
            else:
                ppo_input_norm = ppo_input
            
            # 4. PPO source → zd (U)
            zd, _ = source_unn_policy.ppo_policy.predict(ppo_input_norm, deterministic=True)
            
            # 5. Decoder avec VAE cible → x_rd (Bo_target)
            x_rd = target_base.decode_np(zd, device=device)
            
            # 6. Extraire vitesse: les indices des vitesses angulaires
            # 2DoF: dtheta1 = x_rd[2]*omega_max, dtheta2 = x_rd[3]*omega_max
            # 3DoF: dtheta1 = x_rd[3]*omega_max, dtheta2 = x_rd[4]*omega_max, dtheta3 = x_rd[5]*omega_max
            omega_max = 2.0
            if target_base.state_dim == 6:  # 2DoF
                action = x_rd[2:4] * omega_max
            else:  # 3DoF
                action = x_rd[3:6] * omega_max
            action = np.clip(action, -1.0, 1.0)
            
            obs, _, done, _, info = env.step(action)
        
        if info.get("target_reached", False):
            successes += 1
        
        if (ep + 1) % 20 == 0:
            print(f"    Episode {ep + 1}: {100 * successes / (ep + 1):.1f}%")
    
    rate = 100 * successes / n_episodes
    print(f"    Final success rate: {rate:.1f}%")
    env.close()
    return rate


def test_transfer_2to3(
    source_unn_policy,  # UNNPolicy entraîné sur 2DoF
    target_base: BaseVAE,  # VAE 3DoF pour Bi_target et Bo_target
    device: str = "cpu",
    n_episodes: int = 100,
) -> float:
    """Teste le transfert de la politique 2DoF vers l'environnement 3DoF."""
    return _test_transfer_direction(
        source_unn_policy=source_unn_policy,
        target_base=target_base,
        target_env_class=PushBallEnv_3dof,
        device=device,
        n_episodes=n_episodes,
        direction="2→3",
    )


def test_transfer_3to2(
    source_unn_policy,  # UNNPolicy entraîné sur 3DoF
    target_base: BaseVAE,  # VAE 2DoF pour Bi_target et Bo_target
    device: str = "cpu",
    n_episodes: int = 100,
) -> float:
    """Teste le transfert de la politique 3DoF vers l'environnement 2DoF."""
    return _test_transfer_direction(
        source_unn_policy=source_unn_policy,
        target_base=target_base,
        target_env_class=PushBallEnv_2dof,
        device=device,
        n_episodes=n_episodes,
        direction="3→2",
    )