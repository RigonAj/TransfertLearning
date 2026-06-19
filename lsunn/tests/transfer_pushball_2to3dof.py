"""
Test LS-UNN transfer: 2-DoF PushBall policy → 3-DoF environment.
Utilise le mapper d'action entraîné.

Usage:
    python -m lsunn.tests.transfer_pushball_2to3dof
"""

import sys
import warnings
import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings("ignore", message=".*shared CUDA tensors.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_3dof import PushBallEnv_3dof
from lsunn.bases_vae import BaseVAE, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM
from lsunn.train.train_joint_transfer import ActionMapper

# Dimensions
ARM_OBS_3DOF = 8
ACTION_DIM_2DOF = 2
ACTION_DIM_3DOF = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPISODES = 1000
MAX_STEPS = 150
JOINT_MODEL_DIR = Path("./data/LSUNN/joint_model")


def load_mapper(path: Path, state_dim: int, act_in_dim: int, act_out_dim: int):
    """Charge un mapper entraîné."""
    mapper = ActionMapper(state_dim, act_in_dim, act_out_dim, hidden_dim=512).to(DEVICE)
    mapper.load_state_dict(torch.load(path, map_location=DEVICE))
    mapper.eval()
    return mapper


def main():
    print("\n" + "="*60)
    print("LS-UNN Transfer: 2-DoF → 3-DoF")
    print("="*60)

    # 1. Charger les VAE
    print("\n1. Loading VAE bases...")
    base_2dof = BaseVAE(10, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM).to(DEVICE)
    base_2dof.load_state_dict(torch.load(JOINT_MODEL_DIR / "base_2dof.pt", map_location=DEVICE))
    base_2dof.eval()

    base_3dof = BaseVAE(12, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM).to(DEVICE)
    base_3dof.load_state_dict(torch.load(JOINT_MODEL_DIR / "base_3dof.pt", map_location=DEVICE))
    base_3dof.eval()

    # 2. Charger la politique source (2DoF)
    print("\n2. Loading source PPO policy (2-DoF latent)...")
    ppo_2dof = PPO.load(JOINT_MODEL_DIR / "latent_policy_2dof.zip", device=DEVICE)

    # 3. Charger le normalizer source
    class DummyEnv:
        def __init__(self):
            from gymnasium import spaces
            self.observation_space = spaces.Box(-10, 10, (DEFAULT_LATENT_DIM,), dtype=np.float32)
            self.action_space = spaces.Box(-1, 1, (ACTION_DIM_2DOF,), dtype=np.float32)
        def reset(self): return np.zeros(DEFAULT_LATENT_DIM), {}
        def step(self, a): return np.zeros(DEFAULT_LATENT_DIM), 0, False, False, {}

    dummy = DummyVecEnv([lambda: DummyEnv()])
    vec_norm = VecNormalize.load(JOINT_MODEL_DIR / "vec_normalize_2dof.pkl", venv=dummy)
    vec_norm.training = False
    dummy.close()

    # 4. Charger le mapper 2→3
    print("\n3. Loading action mapper (2DoF → 3DoF)...")
    mapper_path = JOINT_MODEL_DIR / "mapper_2to3.pt"
    if not mapper_path.exists():
        print(f"   ERROR: Mapper not found at {mapper_path}")
        print("   Please run: python -m lsunn.train.train_joint_transfer")
        return

    mapper = load_mapper(mapper_path, ARM_OBS_3DOF, ACTION_DIM_2DOF, ACTION_DIM_3DOF)

    # 5. Créer l'environnement cible (3DoF)
    print("\n4. Creating 3-DoF environment...")
    env = DummyVecEnv([lambda: Monitor(PushBallEnv_3dof(render_mode=None, max_steps=MAX_STEPS))])

    # 6. Test de transfert
    print(f"\n5. Running {NUM_EPISODES} transfer episodes...\n")

    successes = 0
    steps_ok = []
    dist_fail = []
    alignments = []

    for ep in range(NUM_EPISODES):
        obs = env.reset()[0]
        done = False
        step = 0
        info_last = {}

        while not done:
            with torch.no_grad():
                # Encodage de l'observation 3DoF avec VAE 3DoF
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                z = base_3dof.encode(obs_t).cpu().numpy().flatten()

                # Normalisation avec stats de la politique source (2DoF)
                z_norm = vec_norm.normalize_obs(z)

                # Politique source → action 2DoF
                a_2dof, _ = ppo_2dof.predict(z_norm, deterministic=True)

                # Mapper → action 3DoF
                arm_obs = obs[:ARM_OBS_3DOF]
                s_t = torch.tensor(arm_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a_t = torch.tensor(a_2dof, dtype=torch.float32, device=DEVICE)
                if a_t.ndim == 1:
                    a_t = a_t.unsqueeze(0)
                a_3dof = mapper(s_t, a_t).cpu().numpy().flatten()

            obs, _, dones, infos = env.step(a_3dof.reshape(1, -1))
            obs = obs[0]
            step += 1
            done = dones[0]
            info_last = infos[0]

        if info_last.get("target_reached", False):
            successes += 1
            steps_ok.append(step)
        else:
            dist_fail.append(info_last.get("dist_ball_target", float("nan")))

        alignments.append(info_last.get("alignment", 0.0))

        if (ep + 1) % 200 == 0:
            print(f"  Episode {ep+1:4d}/{NUM_EPISODES} | Success rate: {100*successes/(ep+1):5.1f}% | "
                  f"Avg align: {np.mean(alignments[-200:]):.3f}")

    success_rate = 100 * successes / NUM_EPISODES

    print("\n" + "="*70)
    print("LS-UNN Transfer Results: 2-DoF → 3-DoF")
    print("="*70)
    print(f"  Episodes tested      : {NUM_EPISODES}")
    print(f"  Successes            : {successes}")
    print(f"  Success rate         : {success_rate:.1f}%")
    print("-" * 70)

    if steps_ok:
        print(f"  Avg steps (success)  : {np.mean(steps_ok):.1f}")
        print(f"  Steps min/max        : {np.min(steps_ok)} / {np.max(steps_ok)}")

    if dist_fail:
        print(f"  Avg distance (fail)  : {np.mean(dist_fail):.4f} m")
        print(f"  Distance min/max     : {np.min(dist_fail):.4f} / {np.max(dist_fail):.4f} m")

    print(f"  Avg alignment        : {np.mean(alignments):.3f}")
    print("="*70)

    # Sauvegarde des résultats
    results = {
        'success_rate': success_rate,
        'successes': successes,
        'steps_ok': steps_ok,
        'dist_fail': dist_fail,
        'alignments': alignments,
        'n_episodes': NUM_EPISODES,
    }

    results_path = Path("./data/LSUNN/transfer_results_2to3.pkl")
    import pickle
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  Results saved to {results_path}")

    env.close()


if __name__ == "__main__":
    main()