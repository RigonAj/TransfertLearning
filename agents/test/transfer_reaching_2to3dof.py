"""
Transfer Learning — Reaching Task
===================================
Apply the 2-DoF reaching policy to the 3-DoF environment via learned mappers.

Pipelines : identique à pushball mais avec 5D tâche (13D obs 3-DoF, 11D pour 2-DoF).
"""

import numpy as np
import torch
import pickle
import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_reaching_3dof import ReachingEnv_3dof
from envs.env_reaching_2dof import ReachingEnv_2dof

from agents.transfer.eq_mappings_2to3dof import StateMapperMLP, ActionMapperMLP


class TransferPolicy:
    def __init__(self, policy_2dof_path, vecnorm_2dof_path, state_mapper_path, action_mapper_path, device="cpu"):
        self.device = device
        self.policy_2dof = PPO.load(policy_2dof_path, device=device)

        def make_dummy_2dof(): return Monitor(ReachingEnv_2dof(render_mode=None))
        venv = DummyVecEnv([make_dummy_2dof])
        self.vec_norm_2dof = VecNormalize.load(vecnorm_2dof_path, venv=venv)
        self.vec_norm_2dof.training = False
        self.vec_norm_2dof.norm_reward = False
        venv.close()

        self.state_mapper = StateMapperMLP(input_dim=8, output_dim=6).to(device)
        self.state_mapper.load_state_dict(torch.load(state_mapper_path, map_location=device))
        self.state_mapper.eval()

        self.action_mapper = ActionMapperMLP(state_dim=8, action_2dof_dim=2, output_dim=3).to(device)
        self.action_mapper.load_state_dict(torch.load(action_mapper_path, map_location=device))
        self.action_mapper.eval()

        self.policy_2dof_path = policy_2dof_path
        self.vecnorm_2dof_path = vecnorm_2dof_path
        self.state_mapper_path = state_mapper_path
        self.action_mapper_path = action_mapper_path
        print("[Transfer] loaded reaching 2→3")

    @torch.no_grad()
    def predict(self, raw_obs_3dof):
        if raw_obs_3dof.ndim == 1:
            raw_obs_3dof = raw_obs_3dof.reshape(1, -1)
        arm_obs_3dof = raw_obs_3dof[:, :8]
        task_obs_3dof = raw_obs_3dof[:, 8:]  # 5D
        arm_t = torch.tensor(arm_obs_3dof, dtype=torch.float32, device=self.device)
        arm_obs_2dof_equiv = self.state_mapper(arm_t).cpu().numpy()
        full_obs_2dof_equiv = np.concatenate([arm_obs_2dof_equiv, task_obs_3dof], axis=1)  # (1,11)
        obs_2dof_norm = self.vec_norm_2dof.normalize_obs(full_obs_2dof_equiv)
        action_2dof, _ = self.policy_2dof.predict(obs_2dof_norm, deterministic=True)
        a2_t = torch.tensor(action_2dof, dtype=torch.float32, device=self.device)
        action_3dof = self.action_mapper(arm_t, a2_t).cpu().numpy()
        return action_3dof

    def save(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        config = {
            'policy_2dof_path': self.policy_2dof_path,
            'vecnorm_2dof_path': self.vecnorm_2dof_path,
            'state_mapper_path': self.state_mapper_path,
            'action_mapper_path': self.action_mapper_path,
            'device': self.device,
        }
        with open(Path(save_dir) / "transfer_config.pkl", 'wb') as f:
            pickle.dump(config, f)

    @classmethod
    def load_from_config(cls, save_dir, device="cpu"):
        path = Path(save_dir) / "transfer_config.pkl"
        with open(path, 'rb') as f:
            config = pickle.load(f)
        return cls(config['policy_2dof_path'], config['vecnorm_2dof_path'],
                   config['state_mapper_path'], config['action_mapper_path'], device)


def main():
    DEVICE = "cpu"
    NUM_EPISODES = 1000
    run_id_2dof = 1
    save_transfer_id = 1

    POLICY_2DOF_PATH  = f"./models/ppo_reach_2dof_{run_id_2dof}/best_model.zip"
    VECNORM_2DOF_PATH = f"./models/ppo_reach_2dof_{run_id_2dof}/vec_normalize.pkl"
    STATE_MAPPER_PATH  = "./data/transfer_learning/state_mapper_2to3dof.pt"
    ACTION_MAPPER_PATH = "./data/transfer_learning/action_mapper_2to3dof.pt"
    SAVE_DIR = f"./models/ppo_transfer_reaching_2to3_{save_transfer_id}"

    for p in [POLICY_2DOF_PATH, VECNORM_2DOF_PATH, STATE_MAPPER_PATH, ACTION_MAPPER_PATH]:
        if not Path(p).exists():
            print(f"❌  Missing {p}"); return

    policy = TransferPolicy(POLICY_2DOF_PATH, VECNORM_2DOF_PATH, STATE_MAPPER_PATH, ACTION_MAPPER_PATH, DEVICE)

    env = DummyVecEnv([lambda: Monitor(ReachingEnv_3dof(render_mode=None))])

    successes = 0; steps_ok = []; dist_fail = []
    for ep in range(NUM_EPISODES):
        obs = env.reset()
        done = False; step = 0; info_last = {}
        while not done:
            act = policy.predict(obs[0])
            obs, _, dones, infos = env.step(act)
            step += 1
            done = dones[0]
            info_last = infos[0]
        if info_last.get("target_reached", False):
            successes += 1; steps_ok.append(step)
        else:
            dist_fail.append(info_last.get("dist", float("nan")))
        if (ep+1)%200 == 0:
            print(f"  {ep+1}/{NUM_EPISODES}  success rate {100*successes/(ep+1):.1f}%")

    rate = 100*successes/NUM_EPISODES
    print("="*70)
    print(f"  Taux de réussite : {rate:.1f}%")
    if steps_ok: print(f"  Steps moyens : {np.mean(steps_ok):.1f}")
    if dist_fail: print(f"  Dist échec moy : {np.mean(dist_fail):.4f} m")
    print("="*70)
    policy.save(SAVE_DIR)
    env.close()


if __name__ == "__main__":
    main()
