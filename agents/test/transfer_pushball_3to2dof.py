"""
Transfer Learning — PushBall Task
===================================
Apply the 3-DoF pushball policy to the 2-DoF environment via learned mappers.

Pipelines (per timestep) :
  1. raw_obs_2dof        ← 2-DoF environment (10D : 6 arm + 4 task)
  2. arm_obs_2dof        = raw_obs_2dof[:6]
  3. arm_obs_3dof_equiv  = state_mapper(arm_obs_2dof)          # 6 → 8
  4. full_obs_3dof_equiv = concat(arm_obs_3dof_equiv, task_obs_2dof)  # 8 + 4 = 12
  5. obs_3dof_norm       = vec_norm_3dof.normalize_obs(full_obs_3dof_equiv)
  6. action_3dof         = policy_3dof(obs_3dof_norm)          [deterministic]
  7. action_2dof         = action_mapper(arm_obs_2dof, action_3dof)   # 6+3 → 2
  8. obs_2dof, …         = env_2dof.step(action_2dof)
"""

import numpy as np
import torch
import pickle
import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_3dof import PushBallEnv_3dof
from envs.env_pushball_2dof import PushBallEnv_2dof

from agents.transfer.mappings_3to2dof import StateMapperMLP, ActionMapperMLP


class TransferPolicy:
    def __init__(
        self,
        policy_3dof_path: str,
        vecnorm_3dof_path: str,
        state_mapper_path: str,
        action_mapper_path: str,
        device: str = "cpu",
    ):
        self.device = device

        # ---- 3-DoF PPO policy ----
        self.policy_3dof = PPO.load(policy_3dof_path, device=device)

        # ---- 3-DoF VecNormalize (inference only) ----
        def make_dummy_3dof():
            return Monitor(PushBallEnv_3dof(render_mode=None))
        venv = DummyVecEnv([make_dummy_3dof])
        self.vec_norm_3dof = VecNormalize.load(vecnorm_3dof_path, venv=venv)
        self.vec_norm_3dof.training = False
        self.vec_norm_3dof.norm_reward = False
        venv.close()

        # ---- State Mapper: arm_obs_2dof (6) → arm_obs_3dof (8) ----
        self.state_mapper = StateMapperMLP(input_dim=6, output_dim=8).to(device)
        self.state_mapper.load_state_dict(
            torch.load(state_mapper_path, map_location=device)
        )
        self.state_mapper.eval()

        # ---- Action Mapper: (arm_obs_2dof [6], action_3dof [3]) → action_2dof [2] ----
        self.action_mapper = ActionMapperMLP(
            state_dim=6, action_3dof_dim=3, output_dim=2
        ).to(device)
        self.action_mapper.load_state_dict(
            torch.load(action_mapper_path, map_location=device)
        )
        self.action_mapper.eval()

        self.policy_3dof_path = policy_3dof_path
        self.vecnorm_3dof_path = vecnorm_3dof_path
        self.state_mapper_path = state_mapper_path
        self.action_mapper_path = action_mapper_path

        print("[Transfer] All components loaded successfully")
        print(f"  policy_3dof    : {policy_3dof_path}")
        print(f"  vecnorm_3dof   : {vecnorm_3dof_path}")
        print(f"  state_mapper   : {state_mapper_path}")
        print(f"  action_mapper  : {action_mapper_path}")

    @torch.no_grad()
    def predict(self, raw_obs_2dof: np.ndarray) -> np.ndarray:
        if raw_obs_2dof.ndim == 1:
            raw_obs_2dof = raw_obs_2dof.reshape(1, -1)

        # Séparer
        arm_obs_2dof = raw_obs_2dof[:, :6]
        task_obs_2dof = raw_obs_2dof[:, 6:]  # 4D

        # Mapper le bras 2→3
        arm_t = torch.tensor(arm_obs_2dof, dtype=torch.float32, device=self.device)
        arm_obs_3dof_equiv = self.state_mapper(arm_t).cpu().numpy()  # (1,8)

        # Observation complète 3-DoF
        full_obs_3dof_equiv = np.concatenate([arm_obs_3dof_equiv, task_obs_2dof], axis=1)  # (1,12)

        # Normaliser
        obs_3dof_norm = self.vec_norm_3dof.normalize_obs(full_obs_3dof_equiv)

        # Action 3-DoF
        action_3dof, _ = self.policy_3dof.predict(obs_3dof_norm, deterministic=True)

        # Mapper action
        a3_t = torch.tensor(action_3dof, dtype=torch.float32, device=self.device)
        action_2dof = self.action_mapper(arm_t, a3_t).cpu().numpy()  # (1,2)

        return action_2dof

    def save(self, save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        transfer_config = {
            'policy_3dof_path': self.policy_3dof_path,
            'vecnorm_3dof_path': self.vecnorm_3dof_path,
            'state_mapper_path': self.state_mapper_path,
            'action_mapper_path': self.action_mapper_path,
            'device': self.device,
        }
        with open(save_path / "transfer_config.pkl", 'wb') as f:
            pickle.dump(transfer_config, f)

    @classmethod
    def load_from_config(cls, save_dir: str, device: str = "cpu"):
        save_path = Path(save_dir)
        config_path = save_path / "transfer_config.pkl"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {config_path}")
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        return cls(
            policy_3dof_path=config['policy_3dof_path'],
            vecnorm_3dof_path=config['vecnorm_3dof_path'],
            state_mapper_path=config['state_mapper_path'],
            action_mapper_path=config['action_mapper_path'],
            device=device,
        )


def main():
    DEVICE = "cpu"
    NUM_EPISODES = 1000

    run_id_3dof = 1    
    save_transfer_id = 1

    POLICY_3DOF_PATH  = f"./models/ppo_pushball_3dof_{run_id_3dof}/best_model.zip"
    VECNORM_3DOF_PATH = f"./models/ppo_pushball_3dof_{run_id_3dof}/vec_normalize.pkl"
    STATE_MAPPER_PATH  = "./data/transfer_learning/state_mapper_3to2dof.pt"
    ACTION_MAPPER_PATH = "./data/transfer_learning/action_mapper_3to2dof.pt"

    SAVE_DIR = f"./models/ppo_transfer_pushball_3to2_{save_transfer_id}"

    for p in [POLICY_3DOF_PATH, VECNORM_3DOF_PATH, STATE_MAPPER_PATH, ACTION_MAPPER_PATH]:
        if not Path(p).exists():
            print(f"❌  File not found: {p}")
            return

    print("\n" + "="*60)
    print("TRANSFER LEARNING TEST: 3-DoF PushBall → 2-DoF Environment")
    print("="*60)

    policy = TransferPolicy(
        policy_3dof_path=POLICY_3DOF_PATH,
        vecnorm_3dof_path=VECNORM_3DOF_PATH,
        state_mapper_path=STATE_MAPPER_PATH,
        action_mapper_path=ACTION_MAPPER_PATH,
        device=DEVICE,
    )

    def make_env():
        return Monitor(PushBallEnv_2dof(render_mode=None))
    env = DummyVecEnv([make_env])

    successes = 0
    steps_on_success = []
    final_dist_failure = []

    for ep in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        step = 0
        info_last = {}

        while not done:
            action_2dof = policy.predict(obs[0])
            obs, _, dones, infos = env.step(action_2dof)
            step += 1
            done = dones[0]
            info_last = infos[0]

        if info_last.get("target_reached", False):
            successes += 1
            steps_on_success.append(step)
        else:
            final_dist_failure.append(info_last.get("dist_ball_target", float("nan")))

        if (ep + 1) % 200 == 0:
            print(f"  Progress: {ep + 1}/{NUM_EPISODES}  "
                  f"(succès partiel: {100 * successes / (ep + 1):.1f}%)")

    rate = 100.0 * successes / NUM_EPISODES
    print("\n" + "="*70)
    print("RÉSULTATS DU TRANSFERT")
    print("="*70)
    print(f"  Épisodes testés       : {NUM_EPISODES}")
    print(f"  Réussites             : {successes}")
    print(f"  Taux de réussite      : {rate:.1f}%")
    if steps_on_success:
        print(f"  Steps moyens (succès) : {np.mean(steps_on_success):.1f}")
    if final_dist_failure:
        print(f"  Distance moy (échec)  : {np.mean(final_dist_failure):.4f} m")
    print("="*70)

    policy.save(SAVE_DIR)
    env.close()


if __name__ == "__main__":
    main()
