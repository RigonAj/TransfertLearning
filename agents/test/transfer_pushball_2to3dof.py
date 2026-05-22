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

from envs.env_pushball_3dof import PushBallEnv_3dof
from envs.env_pushball_2dof import PushBallEnv_2dof
from agents.transfer.mapper_models import (
    StateMapperMLP,
    ActionMapperMLP,
    project_mapped_arm_state_to_reference,
)


class PushBallTransfer2to3:
    def __init__(self, policy_2dof_path: str, vecnorm_2dof_path: str,
                 mapper_path: str, device="cpu"):
        self.device = device

        self.policy_2dof = PPO.load(policy_2dof_path, device=device)

        # VecNormalize 2DoF
        venv = DummyVecEnv([lambda: Monitor(PushBallEnv_2dof(render_mode=None))])
        self.vec_norm_2dof = VecNormalize.load(vecnorm_2dof_path, venv=venv)
        self.vec_norm_2dof.training = False
        self.vec_norm_2dof.norm_reward = False
        venv.close()

        # Chargement des mappers depuis le fichier unique
        checkpoint = torch.load(mapper_path, map_location=device)
        self.state_mapper = StateMapperMLP(8, 6).to(device)
        self.state_mapper.load_state_dict(checkpoint["state_mapper"])
        self.state_mapper.eval()

        self.action_mapper = ActionMapperMLP(8, 2, 3).to(device)
        self.action_mapper.load_state_dict(checkpoint["action_mapper"])
        self.action_mapper.eval()

        print("✅ PushBall Transfer 2→3 loaded successfully")

    def _extract_temporal_features(self, arm_state_3dof: np.ndarray) -> np.ndarray:
        """
        Extrait les features temporelles (velocities) à partir de l'état courant
        et les combine avec les features spatiales pour former l'entrée du mapper.
        """
        # arm_state_3dof: [θ1, θ2, θ3, dθ1, dθ2, dθ3, eff_x, eff_y] (8 dims)
        
        # Features spatiales (angles et position)
        spatial = arm_state_3dof.copy()  # angles, position
        
        # Features temporelles (velocities) - déjà présentes dans arm_state_3dof
        # mais pour le mapper nous voulons les vel/angles/pos comme features temporelles
        # Nous réutilisons les velocities comme features temporelles
        temporal = arm_state_3dof.copy()
        
        # Concaténer spatial et temporal pour former l'entrée du mapper (16 dims)
        combined = np.concatenate([spatial, temporal], axis=-1)
        return combined

    @torch.no_grad()
    def predict(self, obs_3dof: np.ndarray) -> np.ndarray:
        if obs_3dof.ndim == 1:
            obs_3dof = obs_3dof.reshape(1, -1)

        arm_3 = obs_3dof[:, :8]      # 8D arm state
        task = obs_3dof[:, 8:]       # 4D task for pushball

        # Construire l'entrée combinée spatiale + temporelle pour le state mapper
        # Le mapper attend 16 dims (8 spatial + 8 temporal)
        arm_3_combined = self._extract_temporal_features(arm_3)
        
        arm_2 = self.state_mapper(torch.from_numpy(arm_3_combined).float().to(self.device))
        arm_2 = project_mapped_arm_state_to_reference(
            arm_2.cpu().numpy(),
            reference_arm_state=arm_3,
        )

        full_obs_2 = np.concatenate([arm_2, task], axis=1)
        norm_obs = self.vec_norm_2dof.normalize_obs(full_obs_2)

        act_2, _ = self.policy_2dof.predict(norm_obs, deterministic=True)

        # Pour l'action mapper, on utilise également l'entrée combinée
        act_3 = self.action_mapper(
            torch.from_numpy(arm_3_combined).float().to(self.device),
            torch.from_numpy(act_2).float().to(self.device)
        )
        return act_3.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate direct pushball transfer 2DoF -> 3DoF.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum steps per episode.")
    args = parser.parse_args()

    POLICY_2DOF   = "./models/ppo_pushball_2dof_1/best_model.zip"
    VECNORM_2DOF  = "./models/ppo_pushball_2dof_1/vec_normalize.pkl"
    MAPPER_PATH   = "./data/DIRECT/transfer_2to3_seq.pt"

    transfer = PushBallTransfer2to3(POLICY_2DOF, VECNORM_2DOF, MAPPER_PATH)

    env = DummyVecEnv([lambda: Monitor(PushBallEnv_3dof(render_mode=None))])
    n_episodes = args.episodes
    max_steps = args.max_steps

    successes = 0
    steps_success = []

    for ep in tqdm(range(n_episodes), desc="PushBall 2→3 Transfer Test"):
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
    print(f"\n=== PUSHBALL TRANSFER 2→3 RESULTS ===")
    print(f"Success Rate : {rate:.2f}%  ({successes}/{n_episodes})")
    if steps_success:
        print(f"Avg steps (success) : {np.mean(steps_success):.1f}")
    print("="*50)

    env.close()


if __name__ == "__main__":
    main()