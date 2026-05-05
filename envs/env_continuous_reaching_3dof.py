import numpy as np
from envs.env_reaching_3dof import ReachingEnv_3dof


class Arm3DoFPersistentEnv(ReachingEnv_3dof):
    def __init__(self, render_mode=None, **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
        self._target_reached = False
        self.max_steps = np.inf

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._target_reached = False
        return obs, info

    def step(self, action):
        if self._target_reached:
            obs = self._get_obs()
            reward = 0.1
            self._target_reached = False
            return obs, reward, False, False, {
                "target_reached": True,
                "dist": float(np.linalg.norm(
                    self.forward_kinematics(self.theta1, self.theta2, self.theta3) - self.target
                )),
            }

        obs, reward, terminated, truncated, info = super().step(action)

        if info.get("target_reached", False):
            self._target_reached = True
            terminated = False

        return obs, reward, terminated, truncated, info
