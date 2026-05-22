import argparse
import numpy as np
import os
import torch
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_reaching_2dof import ReachingEnv_2dof
from envs.env_reaching_3dof import ReachingEnv_3dof
from agents.transfer.mapper_models import (
    StateMapperMLP,
    ActionMapperMLP,
    project_mapped_arm_state_to_reference,
)


class ReachingTransfer3to2:
    def __init__(self, policy_3dof_path: str, vecnorm_3dof_path: str,
                 mapper_path: str, device="cpu"):
        self.device = device

        self.policy_3dof = PPO.load(policy_3dof_path, device=device)

        # VecNormalize 3DoF
        venv = DummyVecEnv([lambda: Monitor(ReachingEnv_3dof(render_mode=None))])
        self.vec_norm_3dof = VecNormalize.load(vecnorm_3dof_path, venv=venv)
        self.vec_norm_3dof.training = False
        self.vec_norm_3dof.norm_reward = False
        venv.close()

        # Chargement des mappers depuis le fichier unique
        checkpoint = torch.load(mapper_path, map_location=device)
        self.state_mapper = StateMapperMLP(6, 8).to(device)
        self.state_mapper.load_state_dict(checkpoint["state_mapper"])
        self.state_mapper.eval()

        self.action_mapper = ActionMapperMLP(6, 3, 2).to(device)
        self.action_mapper.load_state_dict(checkpoint["action_mapper"])
        self.action_mapper.eval()

        print("✅ Reaching Transfer 3→2 loaded successfully")

    def _extract_temporal_features(self, arm_state_2dof: np.ndarray) -> np.ndarray:
        """
        Extrait les features temporelles (velocities) à partir de l'état courant
        et les combine avec les features spatiales pour former l'entrée du mapper.
        """
        # arm_state_2dof: [θ1, θ2, dθ1, dθ2, eff_x, eff_y] (6 dims)
        
        # Features spatiales (angles et position)
        spatial = arm_state_2dof.copy()
        
        # Features temporelles (velocities)
        temporal = arm_state_2dof.copy()
        
        # Concaténer spatial et temporal pour former l'entrée du mapper (12 dims)
        combined = np.concatenate([spatial, temporal], axis=-1)
        return combined

    @torch.no_grad()
    def predict(self, obs_2dof: np.ndarray) -> np.ndarray:
        if obs_2dof.ndim == 1:
            obs_2dof = obs_2dof.reshape(1, -1)

        arm_2 = obs_2dof[:, :6]      # 6D arm state
        task = obs_2dof[:, 6:]       # 5D task for reaching

        # Construire l'entrée combinée spatiale + temporelle pour le state mapper
        # Le mapper attend 12 dims (6 spatial + 6 temporal)
        arm_2_combined = self._extract_temporal_features(arm_2)

        arm_3 = self.state_mapper(torch.from_numpy(arm_2_combined).float().to(self.device))
        arm_3 = project_mapped_arm_state_to_reference(
            arm_3.cpu().numpy(),
            reference_arm_state=arm_2,
        )

        full_obs_3 = np.concatenate([arm_3, task], axis=1)
        norm_obs = self.vec_norm_3dof.normalize_obs(full_obs_3)

        act_3, _ = self.policy_3dof.predict(norm_obs, deterministic=True)

        # Pour l'action mapper, on utilise également l'entrée combinée
        act_2 = self.action_mapper(
            torch.from_numpy(arm_2_combined).float().to(self.device),
            torch.from_numpy(act_3).float().to(self.device)
        )
        return act_2.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate direct reaching transfer 3DoF -> 2DoF.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode.")
    args = parser.parse_args()

    POLICY_3DOF   = "./models/ppo_reach_3dof_1/best_model.zip"
    VECNORM_3DOF  = "./models/ppo_reach_3dof_1/vec_normalize.pkl"
    MAPPER_PATH   = "./data/DIRECT/transfer_3to2_seq.pt"

    transfer = ReachingTransfer3to2(POLICY_3DOF, VECNORM_3DOF, MAPPER_PATH)

    env = DummyVecEnv([lambda: Monitor(ReachingEnv_2dof(render_mode=None))])
    n_episodes = args.episodes
    max_steps = args.max_steps

    successes = 0
    steps_success = []

    for ep in tqdm(range(n_episodes), desc="Reaching 3→2 Transfer Test"):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = transfer.predict(obs[0])
            obs, _, dones, infos = env.step([action])
            steps += 1
            done = dones[0]
            info = infos[0]

        if info.get("target_reached", False):
            successes += 1
            steps_success.append(steps)

    rate = successes / n_episodes * 100
    print(f"\n=== REACHING TRANSFER 3→2 RESULTS ===")
    print(f"Success Rate : {rate:.2f}%  ({successes}/{n_episodes})")
    if steps_success:
        print(f"Avg steps (success) : {np.mean(steps_success):.1f}")
    print("="*50)

    env.close()


if __name__ == "__main__":
    main()