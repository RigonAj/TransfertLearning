"""
Test LS-UNN transfer: 3-DoF PushBall policy → 2-DoF environment.
Usage:
    python -m lsunn.test.transfer_pushball_3to2dof
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_2dof import PushBallEnv_2dof
from lsunn.bases_vae import BaseVAE, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM
from lsunn.unn_policy import LatentActionMapper


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPISODES = 1000

    BASES_DIR = Path("./data/lsunn_bases")
    BASE_2DOF_PATH = BASES_DIR / "pushball_bases_r1_2dof.pt"
    BASE_3DOF_PATH = BASES_DIR / "pushball_bases_r2_3dof.pt"

    UNN_MODEL_DIR = Path("./data/LSUNN/lsunn_pushball_3dof_1")
    PPO_PATH = UNN_MODEL_DIR / "unn_ppo_pushball_3dof.zip"
    VECNORM_PATH = UNN_MODEL_DIR / "vec_normalize.pkl"

    MAPPER_PATH = Path("./data/LSUNN/action_mappers/mapper_3to2.pt")

    # Load VAE bases
    print("Loading LS-UNN bases...")
    base_2dof = BaseVAE(state_dim=10, latent_dim=DEFAULT_LATENT_DIM,
                         hidden_dim=DEFAULT_HIDDEN_DIM).to(DEVICE)
    base_2dof.load_state_dict(torch.load(BASE_2DOF_PATH, map_location=DEVICE))
    base_2dof.eval()

    base_3dof = BaseVAE(state_dim=12, latent_dim=DEFAULT_LATENT_DIM,
                         hidden_dim=DEFAULT_HIDDEN_DIM).to(DEVICE)
    base_3dof.load_state_dict(torch.load(BASE_3DOF_PATH, map_location=DEVICE))
    base_3dof.eval()

    # Load PPO policy (trained on 3-DoF latent space)
    print("Loading UNN PPO policy...")
    ppo_policy = PPO.load(PPO_PATH, device=DEVICE)

    # Load VecNormalize
    def make_dummy():
        return Monitor(PushBallEnv_2dof(render_mode=None))
    venv = DummyVecEnv([make_dummy])
    vec_norm = VecNormalize.load(VECNORM_PATH, venv=venv)
    vec_norm.training = False
    vec_norm.norm_reward = False
    venv.close()

    # Action mapper (3-DoF action → 2-DoF action, conditioned on 2-DoF arm state)
    action_mapper = LatentActionMapper(
        state_dim=6,        # 2-DoF arm obs only
        src_action_dim=3,   # 3-DoF action
        tgt_action_dim=2,   # 2-DoF action
    ).to(DEVICE)

    if MAPPER_PATH.exists():
        action_mapper.load_state_dict(torch.load(MAPPER_PATH, map_location=DEVICE))
        print("  Loaded pre-trained action mapper (3→2)")
    else:
        print("  Warning: action mapper not found, using random weights")

    env = DummyVecEnv([lambda: Monitor(PushBallEnv_2dof(render_mode=None, max_steps=150))])

    successes = 0
    steps_ok = []
    dist_fail = []

    print(f"\nRunning {NUM_EPISODES} transfer episodes (LS-UNN 3→2 DoF)...\n")

    for ep in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        step = 0
        info_last = {}

        while not done:
            raw_obs = obs[0]

            with torch.no_grad():
                # Encode 2-DoF observation → latent z using 2-DoF VAE
                z = base_2dof.encode_np(raw_obs)
                z_norm = vec_norm.normalize_obs(z)
                a_3dof, _ = ppo_policy.predict(z_norm, deterministic=True)

                # Map action: 3-DoF → 2-DoF
                arm_obs = raw_obs[:6]
                s_t = torch.tensor(arm_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a_t = torch.tensor(a_3dof, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a_2dof = action_mapper(s_t, a_t).cpu().numpy()[0]

            obs, _, dones, infos = env.step(a_2dof)
            step += 1
            done = dones[0]
            info_last = infos[0]

        if info_last.get("target_reached", False):
            successes += 1
            steps_ok.append(step)
        else:
            dist_fail.append(info_last.get("dist_ball_target", float("nan")))

        if (ep + 1) % 200 == 0:
            print(f"  {ep+1}/{NUM_EPISODES}  success rate {100*successes/(ep+1):.1f}%")

    rate = 100 * successes / NUM_EPISODES
    print("=" * 70)
    print(f"  LS-UNN Transfer 3→2 DoF — Success rate: {rate:.1f}%")
    if steps_ok:
        print(f"  Avg steps (success): {np.mean(steps_ok):.1f}")
    if dist_fail:
        print(f"  Avg dist (failure):  {np.mean(dist_fail):.4f} m")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
