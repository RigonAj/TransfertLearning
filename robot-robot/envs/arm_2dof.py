import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class Arm2DoF(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Taille du bloc d'observation propre au bras (utilisé par les classes filles
    # pour slicer : arm_obs = full_obs[:env.arm_obs_size])
    arm_obs_size = 6

    def __init__(self, render_mode=None):
        super().__init__()

        # Géométrie
        self.l1 = 1.5 ##1.0
        self.l2 = 1.5 ##1.0
        self.max_reach = self.l1 + self.l2          ## 3 # 2.0
        self.eff_radius = 0.05

        # Dynamique générique (utilisée ou non selon la tâche)
        self.omega_max = 2.0      # rad/s
        self.dt        = 0.05     # s

        # Espace d'action commun (2 articulations, commande normalisée [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Espace d'observation de base (6D normalisé, étendu par les tâches)
        # [θ1/π, θ2/π, dθ1/ω_max, dθ2/ω_max, eff_x/r, eff_y/r]  — tout dans [-1, 1]
        obs_high = np.ones(self.arm_obs_size, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        self.render_mode = render_mode

        # État interne
        self.theta1  = 0.0
        self.theta2  = 0.0
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.step_count = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.theta1  = float(self.np_random.uniform(-np.pi, np.pi))
        self.theta2  = float(self.np_random.uniform(-np.pi, np.pi))
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def forward_kinematics(self, t1, t2):
        x = self.l1 * np.cos(t1) + self.l2 * np.cos(t1 + t2)
        y = self.l1 * np.sin(t1) + self.l2 * np.sin(t1 + t2)
        return np.array([x, y], dtype=np.float32)

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Observation bras normalisée (6D).

        Index  Grandeur         Normalisation    Plage garantie
        ─────────────────────────────────────────────────────────
          0    θ1               / π              [-1, 1]
          1    θ2               / π              [-1, 1]
          2    dθ1              / ω_max          [-1, 1]
          3    dθ2              / ω_max          [-1, 1]
          4    eff_x            / max_reach      [-1, 1]
          5    eff_y            / max_reach      [-1, 1]

        Les classes filles appellent super()._get_obs() et
        concatènent leurs dimensions propres à la fin.
        """
        eff = self.forward_kinematics(self.theta1, self.theta2)
        return np.array([
            self.theta1  / np.pi,
            self.theta2  / np.pi,
            self.dtheta1 / self.omega_max,
            self.dtheta2 / self.omega_max,
            eff[0] / self.max_reach,
            eff[1] / self.max_reach,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def get_end_effector_pos(self):
        return self.forward_kinematics(
            self.theta1,
            self.theta2
        )

    # ------------------------------------------------------------------
    def get_end_effector_vel(self):

        J = np.array([
            [
                -self.l1*np.sin(self.theta1)
                -self.l2*np.sin(self.theta1+self.theta2),

                -self.l2*np.sin(self.theta1+self.theta2)
            ],
            [
                self.l1*np.cos(self.theta1)
                +self.l2*np.cos(self.theta1+self.theta2),

                self.l2*np.cos(self.theta1+self.theta2)
            ]
        ])

        qdot = np.array([
            self.dtheta1,
            self.dtheta2
        ])

        return J @ qdot


    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        plt.clf()
        j2  = np.array([self.l1 * np.cos(self.theta1),
                        self.l1 * np.sin(self.theta1)])
        eff = self.forward_kinematics(self.theta1, self.theta2)

        plt.plot([0, j2[0]], [0, j2[1]], 'r-', lw=4, label='Link 1')
        plt.plot([j2[0], eff[0]], [j2[1], eff[1]], 'b-', lw=4, label='Link 2')
        plt.gca().add_patch(plt.Circle(eff, self.eff_radius * 2,
                                       color='red', alpha=0.6))
        plt.xlim(-3.2, 3.2)
        plt.ylim(-3.2, 3.2)
        plt.gca().set_aspect("equal")
        plt.title(f"Step {self.step_count}")
        plt.legend(loc="upper right", fontsize=8)
        plt.pause(0.001)
