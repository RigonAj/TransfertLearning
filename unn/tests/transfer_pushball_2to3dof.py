"""
Transfer UNN policy trained on 2-DoF to 3-DoF environment.
"""

import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_3dof import PushBallEnv_3dof
from unn.bases_unn import CartesianStateEncoder, ActionMapper
from unn.unn_policy import UNNPolicy

def main():
    device = "cpu"
    # Load 2DoF UNN policy (trained on 2DoF)
    encoder = CartesianStateEncoder(max_reach=3.0)
    policy_2to3 = UNNPolicy.load("./data/UNN/unn_pushball_2dof", name="final",
                                 encoder=encoder, device=device)

    # Replace its action mapper with 3DoF action mapper (for target robot)
    mapper_3dof = ActionMapper(n_joints=3).to(device)
    mapper_3dof.load_state_dict(torch.load("./data/UNN/action_mappers/mapper_3dof.pt"))
    policy_2to3.action_mapper = mapper_3dof
    policy_2to3.n_joints = 3
    policy_2to3.arm_obs_size = 8

    env = DummyVecEnv([lambda: Monitor(PushBallEnv_3dof(render_mode=None, max_steps=150))])

    successes = 0
    N_EPISODES = 1000
    for ep in range(N_EPISODES):
        obs = env.reset()[0]
        done = False
        while not done:
            action = policy_2to3.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            done = dones[0]
        if infos[0].get("target_reached", False):
            successes += 1
        if (ep+1) % 200 == 0:
            print(f"{ep+1}/{N_EPISODES} – success rate: {100*successes/(ep+1):.1f}%")

    print(f"\nFinal transfer success rate: {100*successes/N_EPISODES:.1f}%")
    env.close()

if __name__ == "__main__":
    main()
