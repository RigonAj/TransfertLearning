"""
Transfer Learning — PushBall Task
===================================
Apply the 2-DoF pushball policy to the 3-DoF environment via learned mappers.

Pipelines (per timestep) :
  1. raw_obs_3dof        ← 3-DoF environment (12D : 8 arm + 4 task)
  2. arm_obs_3dof        = raw_obs_3dof[:8]
  3. arm_obs_2dof_equiv  = state_mapper(arm_obs_3dof)          # 8 → 6
  4. full_obs_2dof_equiv = concat(arm_obs_2dof_equiv, task_obs_3dof)  # 6 + 4 = 10
  5. obs_2dof_norm       = vec_norm_2dof.normalize_obs(full_obs_2dof_equiv)
  6. action_2dof         = policy_2dof(obs_2dof_norm)          [deterministic]
  7. action_3dof         = action_mapper(arm_obs_3dof, action_2dof)   # 8+2 → 3
  8. obs_3dof, …         = env_3dof.step(action_3dof)
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

# Assurez-vous que les classes de mappeurs sont importées depuis le bon dossier
from agents.transfer.eq_mappings_2to3dof import StateMapperMLP, ActionMapperMLP


class TransferPolicy:
    def __init__(
        self,
        policy_2dof_path: str,
        vecnorm_2dof_path: str,
        state_mapper_path: str,
        action_mapper_path: str,
        device: str = "cpu",
    ):
        self.device = device

        # ---- 2-DoF PPO policy ----
        self.policy_2dof = PPO.load(policy_2dof_path, device=device)

        # ---- 2-DoF VecNormalize (inference only) ----
        def make_dummy_2dof():
            return Monitor(PushBallEnv_2dof(render_mode=None))
        venv = DummyVecEnv([make_dummy_2dof])
        self.vec_norm_2dof = VecNormalize.load(vecnorm_2dof_path, venv=venv)
        self.vec_norm_2dof.training = False
        self.vec_norm_2dof.norm_reward = False
        venv.close()

        # ---- State Mapper: arm_obs_3dof (8) → arm_obs_2dof (6) ----
        self.state_mapper = StateMapperMLP(input_dim=8, output_dim=6).to(device)
        self.state_mapper.load_state_dict(
            torch.load(state_mapper_path, map_location=device)
        )
        self.state_mapper.eval()

        # ---- Action Mapper: (arm_obs_3dof [8], action_2dof [2]) → action_3dof [3] ----
        self.action_mapper = ActionMapperMLP(
            state_dim=8, action_2dof_dim=2, output_dim=3
        ).to(device)
        self.action_mapper.load_state_dict(
            torch.load(action_mapper_path, map_location=device)
        )
        self.action_mapper.eval()

        # Stocker les chemins pour la sauvegarde
        self.policy_2dof_path = policy_2dof_path
        self.vecnorm_2dof_path = vecnorm_2dof_path
        self.state_mapper_path = state_mapper_path
        self.action_mapper_path = action_mapper_path

        print("[Transfer] All components loaded successfully")
        print(f"  policy_2dof    : {policy_2dof_path}")
        print(f"  vecnorm_2dof   : {vecnorm_2dof_path}")
        print(f"  state_mapper   : {state_mapper_path}")
        print(f"  action_mapper  : {action_mapper_path}")

    @torch.no_grad()
    def predict(self, raw_obs_3dof: np.ndarray) -> np.ndarray:
        # raw_obs_3dof : (12,) ou (1,12)
        if raw_obs_3dof.ndim == 1:
            raw_obs_3dof = raw_obs_3dof.reshape(1, -1)

        # 1. Séparer bras et tâche
        arm_obs_3dof = raw_obs_3dof[:, :8]
        task_obs_3dof = raw_obs_3dof[:, 8:]  # 4D (ball_x, ball_y, tgt_x, tgt_y)

        # 2. Mapper le bras 3→2
        arm_t = torch.tensor(arm_obs_3dof, dtype=torch.float32, device=self.device)
        arm_obs_2dof_equiv = self.state_mapper(arm_t).cpu().numpy()  # (1,6)

        # 3. Reconstruire l’observation complète pour le 2-DoF
        full_obs_2dof_equiv = np.concatenate([arm_obs_2dof_equiv, task_obs_3dof], axis=1)  # (1,10)

        # 4. Normaliser avec le VecNormalize 2-DoF
        obs_2dof_norm = self.vec_norm_2dof.normalize_obs(full_obs_2dof_equiv)

        # 5. Action 2-DoF
        action_2dof, _ = self.policy_2dof.predict(obs_2dof_norm, deterministic=True)

        # 6. Mapper l’action (context-aware)
        a2_t = torch.tensor(action_2dof, dtype=torch.float32, device=self.device)
        action_3dof = self.action_mapper(arm_t, a2_t).cpu().numpy()  # (1,3)

        return action_3dof

    def save(self, save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        transfer_config = {
            'policy_2dof_path': self.policy_2dof_path,
            'vecnorm_2dof_path': self.vecnorm_2dof_path,
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
            policy_2dof_path=config['policy_2dof_path'],
            vecnorm_2dof_path=config['vecnorm_2dof_path'],
            state_mapper_path=config['state_mapper_path'],
            action_mapper_path=config['action_mapper_path'],
            device=device,
        )


def main():
    DEVICE = "cpu"
    NUM_EPISODES = 1000

    run_id_2dof = 1  
    save_transfer_id = 1

    POLICY_2DOF_PATH  = f"./models/ppo_pushball_2dof_{run_id_2dof}/best_model.zip"
    VECNORM_2DOF_PATH = f"./models/ppo_pushball_2dof_{run_id_2dof}/vec_normalize.pkl"
    STATE_MAPPER_PATH  = "./data/transfer_learning/state_mapper_2to3dof.pt"
    ACTION_MAPPER_PATH = "./data/transfer_learning/action_mapper_2to3dof.pt"

    SAVE_DIR = f"./models/ppo_transfer_pushball_2to3_{save_transfer_id}"

    for p in [POLICY_2DOF_PATH, VECNORM_2DOF_PATH, STATE_MAPPER_PATH, ACTION_MAPPER_PATH]:
        if not Path(p).exists():
            print(f"❌  File not found: {p}")
            return

    print("\n" + "="*60)
    print("TRANSFER LEARNING TEST: 2-DoF PushBall → 3-DoF Environment")
    print("="*60)

    policy = TransferPolicy(
        policy_2dof_path=POLICY_2DOF_PATH,
        vecnorm_2dof_path=VECNORM_2DOF_PATH,
        state_mapper_path=STATE_MAPPER_PATH,
        action_mapper_path=ACTION_MAPPER_PATH,
        device=DEVICE,
    )

    def make_env():
        return Monitor(PushBallEnv_3dof(render_mode=None))
    env = DummyVecEnv([make_env])

    print(f"\n[2] Running {NUM_EPISODES} episodes...\n")

    successes = 0
    steps_on_success = []
    final_dist_failure = []

    for ep in range(NUM_EPISODES):
        obs = env.reset()          
        done = False
        step = 0
        info_last = {}

        while not done:
            action_3dof = policy.predict(obs[0])
            obs, _, dones, infos = env.step(action_3dof)
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
