import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_reaching_2dof import ReachingEnv_2dof
from envs.env_reaching_3dof import ReachingEnv_3dof
from agents.transfer.mapper_models import StateMapperMLP, ActionMapperMLP


class ReachingTransfer3to2:
    def __init__(self, policy_3dof_path: str, vecnorm_3dof_path: str,
                 state_mapper_path: str, action_mapper_path: str, device="cpu"):
        self.device = device

        self.policy_3dof = PPO.load(policy_3dof_path, device=device)

        # VecNormalize 3DoF
        venv = DummyVecEnv([lambda: Monitor(ReachingEnv_3dof(render_mode=None))])
        self.vec_norm_3dof = VecNormalize.load(vecnorm_3dof_path, venv=venv)
        self.vec_norm_3dof.training = False
        self.vec_norm_3dof.norm_reward = False
        venv.close()

        # Mappers 3→2
        self.state_mapper = StateMapperMLP(6, 8).to(device)      # 2→3 obs
        self.state_mapper.load_state_dict(torch.load(state_mapper_path, map_location=device))
        self.state_mapper.eval()

        self.action_mapper = ActionMapperMLP(8, 3, 2).to(device) # 3obs + 3act → 2act
        self.action_mapper.load_state_dict(torch.load(action_mapper_path, map_location=device))
        self.action_mapper.eval()

        print("✅ Reaching Transfer 3→2 loaded successfully")

    @torch.no_grad()
    def predict(self, obs_2dof: np.ndarray) -> np.ndarray:
        if obs_2dof.ndim == 1:
            obs_2dof = obs_2dof.reshape(1, -1)

        arm_2 = obs_2dof[:, :6]
        task = obs_2dof[:, 6:]   # 5D task for reaching

        # 2DoF arm → 3DoF arm equivalent
        arm_3 = self.state_mapper(torch.from_numpy(arm_2).float().to(self.device))
        arm_3 = arm_3.cpu().numpy()

        # Reconstruct full 3DoF observation
        full_obs_3 = np.concatenate([arm_3, task], axis=1)
        norm_obs = self.vec_norm_3dof.normalize_obs(full_obs_3)

        # Get 3DoF action from policy
        act_3, _ = self.policy_3dof.predict(norm_obs, deterministic=True)

        # Map 3DoF action → 2DoF action
        act_2 = self.action_mapper(
            torch.from_numpy(arm_3).float().to(self.device),
            torch.from_numpy(act_3).float().to(self.device)
        )
        return act_2.cpu().numpy()


def main():
    # ==================== CONFIG ====================
    POLICY_3DOF   = "./models/ppo_reach_3dof_1/best_model.zip"
    VECNORM_3DOF  = "./models/ppo_reach_3dof_1/vec_normalize.pkl"
    STATE_MAPPER  = "./data/DIRECT/transfer_3to2.pt"           # ou state_mapper_2to3dof.pt
    ACTION_MAPPER = "./data/DIRECT/action_mapper_3to2dof.pt"

    transfer = ReachingTransfer3to2(POLICY_3DOF, VECNORM_3DOF, STATE_MAPPER, ACTION_MAPPER)

    env = DummyVecEnv([lambda: Monitor(ReachingEnv_2dof(render_mode=None))])
    n_episodes = 500
    max_steps = 200

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
