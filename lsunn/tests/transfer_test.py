# lsunn/train/transfer_test.py
"""
Tests de transfert bidirectionnel 2DoF ↔ 3DoF.
"""

import torch

from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof
from .constants import ARM_OBS_2DOF, ARM_OBS_3DOF


@torch.no_grad()
def _test_transfer_direction(
    source_policy,
    source_vec_norm,
    target_base,
    target_env_class,
    mapper,
    device: str,
    n_episodes: int,
    direction: str,
) -> float:
    """
    Teste une direction de transfert.

    Protocole :
        1. Encode l'observation cible avec le VAE cible.
        2. Normalise avec les stats de la politique source.
        3. La politique source prédit une action source.
        4. Le mapper traduit l'action source en action cible.
        5. Exécute l'action dans l'environnement cible.

    Returns:
        Taux de succès en pourcentage.
    """
    print(f"\n  Testing transfer {direction}...")

    env      = target_env_class(render_mode=None, max_steps=150)
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False

        while not done:
            # 1. Encodage avec VAE cible
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            z     = target_base.encode(obs_t).detach().cpu().numpy().flatten()

            # 2. Normalisation avec stats source
            z_norm = source_vec_norm.normalize_obs(z)

            # 3. Politique source → action source
            a_src, _ = source_policy.predict(z_norm, deterministic=True)

            # 4. Mapper → action cible
            arm_obs = obs[:ARM_OBS_3DOF] if direction == "2→3" else obs[:ARM_OBS_2DOF]

            s_t = torch.tensor(arm_obs, dtype=torch.float32, device=device).unsqueeze(0)
            a_t = torch.tensor(a_src,  dtype=torch.float32, device=device)
            if a_t.ndim == 1:
                a_t = a_t.unsqueeze(0)

            print(f"    Debug: s_t shape={s_t.shape}, a_t shape={a_t.shape}")
            a_tgt = mapper(s_t, a_t).detach().cpu().numpy().flatten()

            # 5. Step dans l'environnement cible
            obs, _, done, _, info = env.step(a_tgt)

        if info.get("target_reached", False):
            successes += 1

        if (ep + 1) % 20 == 0:
            print(f"    Episode {ep + 1}: {100 * successes / (ep + 1):.1f}%")

    rate = 100 * successes / n_episodes
    print(f"    Final success rate: {rate:.1f}%")
    env.close()
    return rate


@torch.no_grad()
def test_transfer_2to3(
    ppo_2dof,
    vec_norm_2dof,
    base_3dof,
    mapper_2to3,
    device: str = "cpu",
    n_episodes: int = 100,
) -> float:
    """Teste le transfert de la politique 2DoF vers l'environnement 3DoF."""
    return _test_transfer_direction(
        source_policy=ppo_2dof,
        source_vec_norm=vec_norm_2dof,
        target_base=base_3dof,
        target_env_class=PushBallEnv_3dof,
        mapper=mapper_2to3,
        device=device,
        n_episodes=n_episodes,
        direction="2→3",
    )


@torch.no_grad()
def test_transfer_3to2(
    ppo_3dof,
    vec_norm_3dof,
    base_2dof,
    mapper_3to2,
    device: str = "cpu",
    n_episodes: int = 100,
) -> float:
    """Teste le transfert de la politique 3DoF vers l'environnement 2DoF."""
    return _test_transfer_direction(
        source_policy=ppo_3dof,
        source_vec_norm=vec_norm_3dof,
        target_base=base_2dof,
        target_env_class=PushBallEnv_2dof,
        mapper=mapper_3to2,
        device=device,
        n_episodes=n_episodes,
        direction="3→2",
    )
