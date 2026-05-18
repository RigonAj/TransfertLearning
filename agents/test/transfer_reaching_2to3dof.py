"""
Transfer Learning — Reaching Task
===================================
Apply the 2-DoF reaching policy to the 3-DoF environment via learned mappers.

Pipelines :
  1. raw_obs_3dof (13D) ← 3‑DoF env
  2. arm_obs_3dof = raw_obs_3dof[:8]
  3. arm_obs_2dof_equiv = state_mapper(arm_obs_3dof)   # 8 → 6 (mappeur 3→2)
  4. full_obs_2dof_equiv = concat(arm_obs_2dof_equiv, task_obs_3dof)  # 6+5 = 11
  5. obs_2dof_norm = vec_norm_2dof.normalize_obs(...)
  6. action_2dof = policy_2dof(obs_2dof_norm)
  7. action_3dof = action_mapper(arm_obs_2dof_equiv, action_2dof)   # 6+2 → 3
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_reaching_3dof import ReachingEnv_3dof
from envs.env_reaching_2dof import ReachingEnv_2dof

# Mappeurs : state_mapper (8→6) vient de mappings_3to2dof, action_mapper (6+2→3) vient de mappings_2to3dof
from agents.transfer.mappings_3to2dof import StateMapperMLP   # input 8 → output 6
from agents.transfer.mappings_2to3dof import ActionMapperMLP  # state_dim=6, action_2dof_dim=2 → output_dim=3


class TransferPolicy:
    def __init__(self, policy_2dof_path, vecnorm_2dof_path,
                 state_mapper_path, action_mapper_path, device="cpu"):
        self.device = device

        # ---- 2-DoF PPO + VecNormalize ----
        custom_objects = {
            "learning_rate": 0.0003,
            "lr_schedule": lambda _: 0.0003,
            "clip_range": lambda _: 0.2,
        }
        self.policy_2dof = PPO.load(policy_2dof_path, device=device, custom_objects=custom_objects)

        def make_dummy_2dof():
            return Monitor(ReachingEnv_2dof(render_mode=None))
        venv = DummyVecEnv([make_dummy_2dof])
        self.vec_norm_2dof = VecNormalize.load(vecnorm_2dof_path, venv=venv)
        self.vec_norm_2dof.training = False
        self.vec_norm_2dof.norm_reward = False
        venv.close()

        # ---- State mapper : 3-DoF arm (8) → 2-DoF arm (6) ----
        self.state_mapper = StateMapperMLP(input_dim=8, output_dim=6).to(device)
        self.state_mapper.load_state_dict(torch.load(state_mapper_path, map_location=device))
        self.state_mapper.eval()

        # ---- Action mapper : (2-DoF arm (6), action_2dof (2)) → action_3dof (3) ----
        self.action_mapper = ActionMapperMLP(state_dim=6, action_2dof_dim=2, output_dim=3).to(device)
        self.action_mapper.load_state_dict(torch.load(action_mapper_path, map_location=device))
        self.action_mapper.eval()

        self.policy_2dof_path = policy_2dof_path
        self.vecnorm_2dof_path = vecnorm_2dof_path
        self.state_mapper_path = state_mapper_path
        self.action_mapper_path = action_mapper_path
        print("[Transfer] loaded reaching 2→3 (corrected mappers)")

    @torch.no_grad()
    def predict(self, raw_obs_3dof):
        if raw_obs_3dof.ndim == 1:
            raw_obs_3dof = raw_obs_3dof.reshape(1, -1)

        # 1. Séparer partie bras (8D) et tâche (5D)
        arm_obs_3dof = raw_obs_3dof[:, :8]
        task_obs_3dof = raw_obs_3dof[:, 8:]   # 5D (dx,dy,tgt_x,tgt_y,dist)

        # 2. Mapper le bras 3→2
        arm_t = torch.tensor(arm_obs_3dof, dtype=torch.float32, device=self.device)
        arm_obs_2dof_equiv = self.state_mapper(arm_t).cpu().numpy()  # (1,6)

        # 3. Reconstruire l'observation 2‑DoF complète (6 arm + 5 task)
        full_obs_2dof_equiv = np.concatenate([arm_obs_2dof_equiv, task_obs_3dof], axis=1)  # (1,11)

        # 4. Normaliser avec le VecNormalize du 2‑DoF
        obs_2dof_norm = self.vec_norm_2dof.normalize_obs(full_obs_2dof_equiv)

        # 5. Action 2‑DoF
        action_2dof, _ = self.policy_2dof.predict(obs_2dof_norm, deterministic=True)

        # 6. Mapper l'action : utiliser l'état 2‑DoF équivalent (6D) + action_2dof → action_3dof
        arm_2dof_t = torch.tensor(arm_obs_2dof_equiv, dtype=torch.float32, device=self.device)
        a2_t = torch.tensor(action_2dof, dtype=torch.float32, device=self.device)
        action_3dof = self.action_mapper(arm_2dof_t, a2_t).cpu().numpy()  # (1,3)

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

    # Chemins mis à jour : mappeur d'état 3→2 (8→6) et mappeur d'action 2→3 (6+2→3)
    POLICY_2DOF_PATH   = f"./models/ppo_reach_2dof_{run_id_2dof}/best_model.zip"
    VECNORM_2DOF_PATH  = f"./models/ppo_reach_2dof_{run_id_2dof}/vec_normalize.pkl"
    STATE_MAPPER_PATH  = "./data/DIRECT/state_mapper_3to2dof.pt"     # 8 → 6
    ACTION_MAPPER_PATH = "./data/DIRECT/action_mapper_2to3dof.pt"   # 6+2 → 3
    SAVE_DIR = f"./models/ppo_transfer_reaching_2to3_{save_transfer_id}"

    for p in [POLICY_2DOF_PATH, VECNORM_2DOF_PATH, STATE_MAPPER_PATH, ACTION_MAPPER_PATH]:
        if not Path(p).exists():
            print(f"❌  Missing {p}")
            return

    policy = TransferPolicy(POLICY_2DOF_PATH, VECNORM_2DOF_PATH,
                            STATE_MAPPER_PATH, ACTION_MAPPER_PATH, DEVICE)

    env = DummyVecEnv([lambda: Monitor(ReachingEnv_3dof(render_mode=None))])

    successes = 0
    steps_ok = []
    dist_fail = []
    for ep in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        step = 0
        info_last = {}
        while not done:
            act = policy.predict(obs[0])
            obs, _, dones, infos = env.step(act)
            step += 1
            done = dones[0]
            info_last = infos[0]
        if info_last.get("target_reached", False):
            successes += 1
            steps_ok.append(step)
        else:
            dist_fail.append(info_last.get("dist", float("nan")))
        if (ep+1) % 200 == 0:
            print(f"  {ep+1}/{NUM_EPISODES}  success rate {100*successes/(ep+1):.1f}%")

    rate = 100 * successes / NUM_EPISODES
    print("="*70)
    print(f"  Taux de réussite : {rate:.1f}%")
    if steps_ok:
        print(f"  Steps moyens (succès) : {np.mean(steps_ok):.1f}")
    if dist_fail:
        print(f"  Distance échec moyenne : {np.mean(dist_fail):.4f} m")
    print("="*70)

    policy.save(SAVE_DIR)
    env.close()


if __name__ == "__main__":
    main()
