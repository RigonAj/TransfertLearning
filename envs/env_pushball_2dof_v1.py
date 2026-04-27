"""
PushBallEnv_2dof

Observation (10D) — mapping 2D inspiré du Pusher MuJoCo :
  [0-1] theta1, theta2        — angles articulaires (rad)
  [2-3] dtheta1, dtheta2      — vitesses angulaires RÉELLES post-clip (rad/s)
  [4-5] eff_x, eff_y          — position effecteur (≡ tips_arm du Pusher MuJoCo) 
  [6-7] ball_x, ball_y        — position balle (m)
  [8-9] target_x, target_y    — position cible (m)

"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class PushBallEnv_2dof(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # --- Géométrie bras ---
        self.l1 = 1.0
        self.l2 = 1.0
        self.max_reach = self.l1 + self.l2          # 2.0 m
        self.theta_max = np.pi                       # ±π rad
        self.theta_min = -self.theta_max

        # --- Dynamique ---
        self.omega_max = 2.0      # rad/s max
        self.dt        = 0.05     # s / step
        self.max_steps = 200  

        # --- Physique contact ---
        self.eff_radius        = 0.05
        self.ball_radius       = 0.10
        self.contact_threshold = self.eff_radius + self.ball_radius

        # --- Seuil de succès ---
        self.epsilon = 0.10       # 10 cm

        # --- Poids reward ---
        self.w_dist  = 1.0
        self.w_near  = 0.6
        self.w_ctrl  = 0.1
        self.bonus_success = 30.0

        # --- Espace d'action ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # --- Observation (10D) ---
        r = self.max_reach
        obs_high = np.array([
            self.theta_max, self.theta_max,   # angles
            self.omega_max, self.omega_max,   # vitesses réelles
            r,              r,                # eff x, y
            r * 1.1,        r * 1.1,          # ball x, y
            r,              r,                # target x, y
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        self.render_mode = render_mode

        # état interne
        self.theta1  = 0.0
        self.theta2  = 0.0
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.ball    = np.zeros(2)
        self.target  = np.zeros(2)
        self.step_count = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.theta1  = float(self.np_random.uniform(self.theta_min, self.theta_max))
        self.theta2  = float(self.np_random.uniform(self.theta_min, self.theta_max))
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0

        # Cible dans le workspace (couronne 0.4–0.9 × max_reach)
        r_t = float(self.np_random.uniform(0.4 * self.max_reach, 0.9 * self.max_reach))
        a_t = float(self.np_random.uniform(-np.pi, np.pi))
        self.target = np.array([r_t * np.cos(a_t), r_t * np.sin(a_t)])

        # Balle à ≥ 0.3 m de la cible
        for _ in range(1000):
            r_b = float(self.np_random.uniform(0.3 * self.max_reach, 0.9 * self.max_reach))
            a_b = float(self.np_random.uniform(-np.pi, np.pi))
            candidate = np.array([r_b * np.cos(a_b), r_b * np.sin(a_b)])
            if np.linalg.norm(candidate - self.target) >= 0.3:
                self.ball = candidate
                break
        else:
            self.ball = self.target + np.array([0.4, 0.0])

        # Distances previous
        eff = self.forward_kinematics(self.theta1, self.theta2)
        self.prev_dist_ball_target = float(np.linalg.norm(self.ball - self.target))
        self.prev_dist_eff_ball = float(np.linalg.norm(eff - self.ball))
        
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)
        action_scaled = action * 0.05 
        
        # --- Cinématique : vitesse RÉELLE après clip de butée ---
        # (nul si joint bloqué → signal clair à la politique)
        new_theta1 = np.clip(
            self.theta1 + float(action[0]) * self.omega_max * self.dt,
            self.theta_min, self.theta_max
        )
        new_theta2 = np.clip(
            self.theta2 + float(action[1]) * self.omega_max * self.dt,
            self.theta_min, self.theta_max
        )
        self.dtheta1 = (new_theta1 - self.theta1) / self.dt
        self.dtheta2 = (new_theta2 - self.theta2) / self.dt
        
        # Pénalité limites articulaires
        at_limit = (
            float(abs(new_theta1 - self.theta1) < 1e-6 and abs(action_scaled[0]) > 0.01) +
            float(abs(new_theta2 - self.theta2) < 1e-6 and abs(action_scaled[1]) > 0.01)
        )
        
        # Forward Kinematics
        self.theta1  = new_theta1
        self.theta2  = new_theta2

        eff = self.forward_kinematics(self.theta1, self.theta2)

        # --- Contact effecteur → balle ---
        vec_eff_ball = self.ball - eff
        contact_dist = float(np.linalg.norm(vec_eff_ball))
        if contact_dist < self.contact_threshold and contact_dist > 1e-6:
            normal    = vec_eff_ball / contact_dist
            self.ball = self.ball + normal * (self.contact_threshold - contact_dist)

        # --- Distances ---
        dist_ball_target = float(np.linalg.norm(self.ball - self.target))
        dist_eff_ball = float(np.linalg.norm(self.ball - eff))

        if dist_eff_ball < 0.2 : 
            dist_eff_ball = 0.1
            

        # --- Alignement effecteur → balle → cible (info) ---
        v_eb = self.ball - eff
        v_bt = self.target - self.ball
        if np.linalg.norm(v_eb) > 1e-3 and np.linalg.norm(v_bt) > 1e-3:
            alignment = float(np.dot(v_eb / np.linalg.norm(v_eb),
                                     v_bt / np.linalg.norm(v_bt)))
        else:
            alignment = 0.0

        # --- Reward ---
        reward_near = -self.w_near * dist_eff_ball
        reward_ctrl = -self.w_ctrl * float(np.sum(np.square(action)))
        
        # REWARD
        reward = reward_near + reward_ctrl
        
        # Progress ball target
        info_progress_ball_target = self.prev_dist_ball_target - dist_ball_target
        progress_ball_target = self.prev_dist_ball_target - dist_ball_target
        if progress_ball_target > 0 :
            reward += 50 * progress_ball_target
        
        # Alignement
        reward += 0.4 * alignment 
        
        # Pénalité limites articulaires
        reward -= 0.5 * at_limit

        
        success  = dist_ball_target < self.epsilon
        ball_out = float(np.linalg.norm(self.ball)) > self.max_reach * 1.05

        if success:
            reward += self.bonus_success
        if ball_out:
            reward -= 5.0

        # Mise à jour des distances pour le prochain step
        self.prev_dist_ball_target = dist_ball_target
        self.prev_dist_eff_ball = dist_eff_ball


        terminated = success or ball_out
        truncated  = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "target_reached":   success,
            "dist_ball_target": dist_ball_target,
            "dist_eff_ball":    dist_eff_ball,
            "alignment":        alignment,
            "progress_ball_target": info_progress_ball_target,
        }

    # ------------------------------------------------------------------
    def _get_obs(self):
        eff = self.forward_kinematics(self.theta1, self.theta2)
        return np.array([
            self.theta1,  self.theta2,
            self.dtheta1, self.dtheta2,
            eff[0],       eff[1],        # ← position effecteur (tips_arm Pusher)
            self.ball[0], self.ball[1],
            self.target[0], self.target[1],
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def forward_kinematics(self, t1, t2):
        x = self.l1 * np.cos(t1) + self.l2 * np.cos(t1 + t2)
        y = self.l1 * np.sin(t1) + self.l2 * np.sin(t1 + t2)
        return np.array([x, y])

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return

        plt.clf()
        j2  = np.array([self.l1 * np.cos(self.theta1), self.l1 * np.sin(self.theta1)])
        eff = self.forward_kinematics(self.theta1, self.theta2)

        plt.plot([0, j2[0]], [0, j2[1]], 'r-', lw=4, label='Link 1')
        plt.plot([j2[0], eff[0]], [j2[1], eff[1]], 'b-', lw=4, label='Link 2')
        plt.gca().add_patch(plt.Circle(eff, self.eff_radius * 2, color='red', alpha=0.6))

        ball_out   = float(np.linalg.norm(self.ball)) > self.max_reach * 1.05
        ball_color = 'red' if ball_out else 'dodgerblue'
        plt.gca().add_patch(plt.Circle(self.ball, self.ball_radius * 1.5,
                                       color=ball_color, alpha=0.6))

        plt.plot(self.target[0], self.target[1], 'o', markersize=18, label='Target')
        plt.gca().add_patch(plt.Circle(self.target, self.epsilon,
                                       color='green', fill=False, linestyle='--', lw=1.5))

        vec = self.target - self.ball
        if np.linalg.norm(vec) > 1e-3:
            v = vec / np.linalg.norm(vec) * 0.3
            plt.arrow(self.ball[0], self.ball[1], v[0], v[1],
                      head_width=0.08, head_length=0.04, fc='green', ec='green', alpha=0.4)

        dist_ball_target = np.linalg.norm(self.ball - self.target)
        plt.xlim(-2.5, 2.5); plt.ylim(-2.5, 2.5)
        plt.gca().set_aspect("equal")
        plt.title(f"Step {self.step_count} | d(ball,tgt)={dist_ball_target:.3f} m"
                  + (" | ⚠️ OUT" if ball_out else ""))
        plt.legend(loc="upper right", fontsize=8)
        plt.pause(0.001)
