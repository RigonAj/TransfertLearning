import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_3dof import PushBallEnv_3dof
from envs.env_pushball_2dof import PushBallEnv_2dof
from agents.transfer.mapper_models import StateMapperMLP, ActionMapperMLP


class PushBallTransfer2to3:
    def __init__(self, policy_2dof_path: str, vecnorm_2dof_path: str,
                 state_mapper_path: str, action_mapper_path: str, device="cpu"):
        self.device = device

        self.policy_2dof = PPO.load(policy_2dof_path, device=device)

        # VecNormalize 2DoF
        venv = DummyVecEnv([lambda: Monitor(PushBallEnv_2dof(render_mode=None))])
        self.vec_norm_2dof = VecNormalize.load(vecnorm_2dof_path, venv=venv)
        self.vec_norm_2dof.training = False
        self.vec_norm_2dof.norm_reward = False
        venv.close()

        # Mappers
        self.state_mapper = StateMapperMLP(8, 6).to(device)
        self.state_mapper.load_state_dict(torch.load(state_mapper_path, map_location=device))
        self.state_mapper.eval()

        self.action_mapper = ActionMapperMLP(6, 2, 3).to(device)
        self.action_mapper.load_state_dict(torch.load(action_mapper_path, map_location=device))
        self.action_mapper.eval()

        print("✅ PushBall Transfer 2→3 loaded successfully")

    @torch.no_grad()
    def predict(self, obs_3dof: np.ndarray) -> np.ndarray:
        if obs_3dof.ndim == 1:
            obs_3dof = obs_3dof.reshape(1, -1)

        arm_3 = obs_3dof[:, :8]
        task = obs_3dof[:, 8:]

        # 3DoF arm → 2DoF arm equivalent
        arm_2 = self.state_mapper(torch.from_numpy(arm_3).float().to(self.device))
        arm_2 = arm_2.cpu().numpy()

        # Reconstruct full 2DoF observation
        full_obs_2 = np.concatenate([arm_2, task], axis=1)
        norm_obs = self.vec_norm_2dof.normalize_obs(full_obs_2)

        # Get 2DoF action
        act_2, _ = self.policy_2dof.predict(norm_obs, deterministic=True)

        # Map to 3DoF action
        act_3 = self.action_mapper(
            torch.from_numpy(arm_2).float().to(self.device),
            torch.from_numpy(act_2).float().to(self.device)
        )
        return act_3.cpu().numpy()


def main():
    # ==================== CONFIG ====================
    POLICY_2DOF   = "./models/ppo_pushball_2dof_1/best_model.zip"
    VECNORM_2DOF  = "./models/ppo_pushball_2dof_1/vec_normalize.pkl"
    STATE_MAPPER  = "./data/DIRECT/transfer_2to3.pt"
    ACTION_MAPPER = "./data/DIRECT/action_mapper_2to3dof.pt"   # ou transfer_2to3.pt si tu as tout dedans

    transfer = PushBallTransfer2to3(POLICY_2DOF, VECNORM_2DOF, STATE_MAPPER, ACTION_MAPPER)

    env = DummyVecEnv([lambda: Monitor(PushBallEnv_3dof(render_mode=None))])
    n_episodes = 500
    max_steps = 400

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
