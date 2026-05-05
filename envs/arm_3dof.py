import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class Arm3DoF(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Taille du bloc d'observation propre au bras (utilisé par les classes filles
    # pour slicer : arm_obs = full_obs[:env.arm_obs_size])
    arm_obs_size = 8

    def __init__(self, render_mode=None):
        super().__init__()

        # Géométrie
        self.l1 = 1.0
        self.l2 = 1.0
        self.l3 = 1.0
        self.max_reach = self.l1 + self.l2 + self.l3   # 3.0
        self.theta_min = -np.pi
        self.theta_max =  np.pi
        self.eff_radius = 0.05

        self.omega_max = 2.0
        self.dt        = 0.05

        # Espace d'action (3 articulations)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Espace d'observation de base (8D normalisé, étendu par les tâches)
        # [θ1/π, θ2/π, θ3/π, dθ1/ω_max, dθ2/ω_max, dθ3/ω_max, eff_x/r, eff_y/r]
        obs_high = np.ones(self.arm_obs_size, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        self.render_mode = render_mode

        self.theta1  = 0.0
        self.theta2  = 0.0
        self.theta3  = 0.0
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.dtheta3 = 0.0
        self.step_count = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.theta1  = float(self.np_random.uniform(self.theta_min, self.theta_max))
        self.theta2  = float(self.np_random.uniform(self.theta_min, self.theta_max))
        self.theta3  = float(self.np_random.uniform(self.theta_min, self.theta_max))
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.dtheta3 = 0.0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def forward_kinematics(self, t1, t2, t3):
        x = (self.l1 * np.cos(t1)
             + self.l2 * np.cos(t1 + t2)
             + self.l3 * np.cos(t1 + t2 + t3))
        y = (self.l1 * np.sin(t1)
             + self.l2 * np.sin(t1 + t2)
             + self.l3 * np.sin(t1 + t2 + t3))
        return np.array([x, y], dtype=np.float32)

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Observation bras normalisée (8D).

        Index  Grandeur         Normalisation    Plage garantie
        ─────────────────────────────────────────────────────────
          0    θ1               / π              [-1, 1]
          1    θ2               / π              [-1, 1]
          2    θ3               / π              [-1, 1]
          3    dθ1              / ω_max          [-1, 1]
          4    dθ2              / ω_max          [-1, 1]
          5    dθ3              / ω_max          [-1, 1]
          6    eff_x            / max_reach      [-1, 1]
          7    eff_y            / max_reach      [-1, 1]

        Les classes filles appellent super()._get_obs() et
        concatènent leurs dimensions propres à la fin.
        """
        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)
        return np.array([
            self.theta1  / np.pi,
            self.theta2  / np.pi,
            self.theta3  / np.pi,
            self.dtheta1 / self.omega_max,
            self.dtheta2 / self.omega_max,
            self.dtheta3 / self.omega_max,
            eff[0] / self.max_reach,
            eff[1] / self.max_reach,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        plt.clf()

        j1  = np.array([self.l1 * np.cos(self.theta1),
                        self.l1 * np.sin(self.theta1)])
        j2  = j1 + np.array([self.l2 * np.cos(self.theta1 + self.theta2),
                              self.l2 * np.sin(self.theta1 + self.theta2)])
        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)

        plt.plot([0, j1[0]], [0, j1[1]], 'r-', lw=4, label='Link 1')
        plt.plot([j1[0], j2[0]], [j1[1], j2[1]], 'b-', lw=4, label='Link 2')
        plt.plot([j2[0], eff[0]], [j2[1], eff[1]], 'm-', lw=4, label='Link 3')
        plt.gca().add_patch(plt.Circle(eff, self.eff_radius * 2,
                                       color='red', alpha=0.6))

        plt.xlim(-3.2, 3.2)
        plt.ylim(-3.2, 3.2)
        plt.gca().set_aspect("equal")
        plt.title(f"Step {self.step_count}")
        plt.legend(loc="upper right", fontsize=8)
        plt.pause(0.001)
