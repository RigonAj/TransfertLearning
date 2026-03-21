import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Arm2DoFEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    ALPHA        = 10.0   # poids de la progression
    LAMBDA_CTRL  = 0.05   # pénalité action
    R_SUCCESS    = 1.0    # bonus succès
    EPSILON      = 0.05   # seuil succès (m)
    DELTA_MAX    = 0.1    # incrément max par step (rad)

    def __init__(self, render_mode=None,
                 alpha=ALPHA,
                 lambda_ctrl=LAMBDA_CTRL,
                 r_success=R_SUCCESS,
                 epsilon=EPSILON,
                 delta_max=DELTA_MAX):
        super().__init__()

        self.l1 = 1.0
        self.l2 = 1.0
        self.max_reach = self.l1 + self.l2

        self.theta_min = -np.pi
        self.theta_max =  np.pi

        self.alpha       = alpha
        self.lambda_ctrl = lambda_ctrl
        self.r_success   = r_success
        self.epsilon     = epsilon
        self.delta_max   = delta_max

        self.max_steps   = 200
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # [q1/pi, q2/pi, dx, dy, eff_x, eff_y, tgt_x, tgt_y]
        obs_low  = np.array([-1, -1, -2, -2, -2, -2, -2, -2], dtype=np.float32)
        obs_high = np.array([ 1,  1,  2,  2,  2,  2,  2,  2], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.theta1     = 0.0
        self.theta2     = 0.0
        self.target     = np.zeros(2, dtype=np.float32)
        self.prev_dist  = 0.0
        self.step_count = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.theta1 = self.np_random.uniform(-np.pi, np.pi)
        self.theta2 = self.np_random.uniform(-np.pi, np.pi)

        radius = self.np_random.uniform(0.1, self.max_reach)
        angle  = self.np_random.uniform(-np.pi, np.pi)
        self.target = np.array(
            [radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32
        )

        eff = self.forward_kinematics(self.theta1, self.theta2)
        self.prev_dist = float(np.linalg.norm(eff - self.target))

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1

        delta = np.clip(action, -1.0, 1.0) * self.delta_max
        self.theta1 = np.clip(self.theta1 + delta[0], self.theta_min, self.theta_max)
        self.theta2 = np.clip(self.theta2 + delta[1], self.theta_min, self.theta_max)

        eff  = self.forward_kinematics(self.theta1, self.theta2)
        dist = float(np.linalg.norm(eff - self.target))

        # --- Reward ---
        # Progression : signal dense, nul si bloqué, négatif si on s'éloigne
        progress = self.prev_dist - dist
        reward   = self.alpha * progress

        # Régularisation action : pénalise les grands incréments
        reward -= self.lambda_ctrl * float(np.dot(action, action))

        # Bonus succès
        success = dist < self.epsilon
        if success:
            reward += self.r_success

        self.prev_dist = dist
        terminated = success
        truncated  = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "target_reached": success,
            "dist": dist,
        }

    # ------------------------------------------------------------------
    def _get_obs(self):
        eff    = self.forward_kinematics(self.theta1, self.theta2)
        dx, dy = self.target - eff
        return np.array([
            self.theta1 / np.pi,
            self.theta2 / np.pi,
            dx, dy,
            eff[0], eff[1],
            self.target[0], self.target[1],
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def forward_kinematics(self, t1, t2):
        x = self.l1 * np.cos(t1) + self.l2 * np.cos(t1 + t2)
        y = self.l1 * np.sin(t1) + self.l2 * np.sin(t1 + t2)
        return np.array([x, y], dtype=np.float32)

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        import matplotlib.pyplot as plt
        plt.clf()

        j1  = np.array([self.l1 * np.cos(self.theta1),
                         self.l1 * np.sin(self.theta1)])
        eff  = self.forward_kinematics(self.theta1, self.theta2)
        dist = np.linalg.norm(eff - self.target)

        plt.plot([0,     j1[0]],  [0,     j1[1]],  'r-', lw=4)
        plt.plot([j1[0], eff[0]], [j1[1], eff[1]], 'b-', lw=4)

        color = 'lime' if dist < self.epsilon else 'red'
        plt.plot(eff[0],         eff[1],         'o', color=color, markersize=10, label="End-effector")
        plt.plot(self.target[0], self.target[1], 'go', markersize=12,             label="Target")

        plt.xlim(-2.2, 2.2)
        plt.ylim(-2.2, 2.2)
        plt.gca().set_aspect("equal")
        plt.title(
            f"Step {self.step_count}  |  d={dist:.3f}m  |  "
            f"θ1={np.degrees(self.theta1):.0f}°  θ2={np.degrees(self.theta2):.0f}°"
        )
        plt.legend(loc="upper left")
        plt.pause(0.001)
