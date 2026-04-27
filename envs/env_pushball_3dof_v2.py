import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class PushBallEnv_3dof(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # Longueurs des segments
        self.l1 = 1.0
        self.l2 = 1.0
        self.l3 = 1.0
        self.max_reach = self.l1 + self.l2 + self.l3

        self.theta_min = -np.pi
        self.theta_max = np.pi

        # Rayons physiques
        self.eff_radius = 0.05
        self.ball_radius = 0.10

        # Action : vitesses angulaires 3 articulations
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Observation : 3 angles + eff(2) + target(2) + ball(2)
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -3, -3, -3, -3, -3, -3], dtype=np.float32),
            high=np.array([1, 1, 1, 3, 3, 3, 3, 3, 3], dtype=np.float32),
            dtype=np.float32
        )

        self.theta1 = 0.0
        self.theta2 = 0.0
        self.theta3 = 0.0

        self.target = np.zeros(2, dtype=np.float32)
        self.ball = np.zeros(2, dtype=np.float32)

        self.step_count = 0
        self.max_steps = 200
        self.render_mode = render_mode
        
        self.reward_value = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.theta1 = self.np_random.uniform(-np.pi, np.pi)
        self.theta2 = self.np_random.uniform(-np.pi, np.pi)
        self.theta3 = self.np_random.uniform(-np.pi, np.pi)

        # Cible dans workspace
        '''radius_t = self.np_random.uniform(0.1, self.max_reach)
        angle_t = self.np_random.uniform(-np.pi, np.pi)
        self.target = np.array(
            [radius_t * np.cos(angle_t), radius_t * np.sin(angle_t)],
            dtype=np.float32
        )'''
        self.target = np.array([1.5, -1.5], dtype=np.float32)
        '''
        # Balle dans workspace
        radius_b = self.np_random.uniform(0.1, 0.9 * self.max_reach)
        angle_b = self.np_random.uniform(-np.pi, np.pi)
        self.ball = np.array(
            [radius_b * np.cos(angle_b), radius_b * np.sin(angle_b)],
            dtype=np.float32
        )
        '''
        # Balle position initiale fixe
#        self.ball = np.array([1.0, 1.0], dtype=np.float32)
        aaa = self.np_random.uniform(1.0, 1.6)
        self.ball = np.array([aaa, aaa], dtype=np.float32)
        
        # Distances previous
        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)
        self.prev_dist_ball_target = float(np.linalg.norm(self.ball - self.target))
        self.prev_dist_eff_ball = float(np.linalg.norm(eff - self.ball))
        
        return self._get_obs(), {}
        
        
    def step(self, action):
        self.step_count += 1
        action_scaled = action * 0.05 

        new_theta1 = np.clip(self.theta1 + action_scaled[0], self.theta_min, self.theta_max)
        new_theta2 = np.clip(self.theta2 + action_scaled[1], self.theta_min, self.theta_max)
        new_theta3 = np.clip(self.theta3 + action_scaled[2], self.theta_min, self.theta_max)
        
        # Limites articulaires
        at_limit = (
            float(abs(new_theta1 - self.theta1) < 1e-6 and abs(action_scaled[0]) > 0.01) +
            float(abs(new_theta2 - self.theta2) < 1e-6 and abs(action_scaled[1]) > 0.01) +
            float(abs(new_theta3 - self.theta3) < 1e-6 and abs(action_scaled[2]) > 0.01)
        )
        
        # Forward Kinematics
        self.theta1  = new_theta1
        self.theta2  = new_theta2
        self.theta3  = new_theta3
        
        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)
        
        # Distances
        dist_eff_ball = np.linalg.norm(eff - self.ball)
        dist_ball_target = np.linalg.norm(self.ball - self.target)

        # Collision eff / balle
        vec = self.ball - eff
        dist = np.linalg.norm(vec)
        min_dist = self.eff_radius + self.ball_radius

        if dist < min_dist and dist > 1e-8:
            push_vec = vec / dist * (min_dist - dist)
            self.ball += push_vec
        
        # Alignement effecteur → balle → cible
        v_eb = self.ball - eff
        v_bt = self.target - self.ball
        if np.linalg.norm(v_eb) > 1e-3 and np.linalg.norm(v_bt) > 1e-3:
            alignment = float(np.dot(v_eb / np.linalg.norm(v_eb),
                                     v_bt / np.linalg.norm(v_bt)))
        else:
            alignment = 0.0
        
        
        reward = 0
        
        # Progression balle → cible
        progress_ball_target = self.prev_dist_ball_target - dist_ball_target
        reward += 2.0 * np.exp(-8 * dist_ball_target)
        reward += 15 * progress_ball_target 
        
        # Progression eff → balle
        progress_eff_ball = self.prev_dist_eff_ball - dist_eff_ball
        reward += 20.0 * progress_eff_ball  
        
        # Alignement
        reward += 0.8 * alignment  
        
        # Pénalité limites articulaires
        reward -= 0.5 * at_limit

        success = np.linalg.norm(self.ball - self.target) < 0.1
        if success :
            reward += 500 
        
        # Mise à jour des distances pour le prochain step
        self.prev_dist_ball_target = np.linalg.norm(self.ball - self.target)
        self.prev_dist_eff_ball = np.linalg.norm(eff - self.ball)
        
        truncated = self.step_count >= self.max_steps
        self.reward_value = reward

        return self._get_obs(), reward, False, truncated, {"target_reached": success}

    def _get_obs(self):
        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)

        return np.array([
            self.theta1 / np.pi,
            self.theta2 / np.pi,
            self.theta3 / np.pi,
            eff[0], eff[1],
            self.target[0], self.target[1],
            self.ball[0], self.ball[1]
        ], dtype=np.float32)

    def forward_kinematics(self, t1, t2, t3):
        x = (
            self.l1 * np.cos(t1)
            + self.l2 * np.cos(t1 + t2)
            + self.l3 * np.cos(t1 + t2 + t3)
        )
        y = (
            self.l1 * np.sin(t1)
            + self.l2 * np.sin(t1 + t2)
            + self.l3 * np.sin(t1 + t2 + t3)
        )
        return np.array([x, y], dtype=np.float32)

    def render(self):
        if self.render_mode != "human":
            return

        plt.clf()

        # Joints intermédiaires
        j1 = np.array([
            self.l1 * np.cos(self.theta1),
            self.l1 * np.sin(self.theta1)
        ])

        j2 = j1 + np.array([
            self.l2 * np.cos(self.theta1 + self.theta2),
            self.l2 * np.sin(self.theta1 + self.theta2)
        ])

        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)

        # Bras
        plt.plot([0, j1[0]], [0, j1[1]], 'r-', lw=4, label='Link 1')
        plt.plot([j1[0], j2[0]], [j1[1], j2[1]], 'b-', lw=4, label='Link 2')
        plt.plot([j2[0], eff[0]], [j2[1], eff[1]], 'm-', lw=4, label='Link 3')

        # Collision
        vec = self.ball - eff
        dist = np.linalg.norm(vec)
        collision = dist < (self.eff_radius + self.ball_radius)

        # Effecteur
        eff_color = 'red' #if not collision else 'orange'
        eff_circle = plt.Circle(eff, self.eff_radius * 2, color=eff_color, alpha=0.6)
        plt.gca().add_patch(eff_circle)

        # Balle
        ball_color = 'blue' #if not collision else 'cyan'
        ball_circle = plt.Circle(self.ball, self.ball_radius * 1.1, color=ball_color, alpha=0.6) #, label='Ball')
        plt.gca().add_patch(ball_circle)

        # Cible
        plt.plot(self.target[0], self.target[1], 'go', markersize=12, label='Target')
        
        # Fleche
        vec = self.target - self.ball
        if np.linalg.norm(vec) > 1e-3:
            v = vec / np.linalg.norm(vec) * 0.3
            plt.arrow(self.ball[0], self.ball[1], v[0], v[1],
                      head_width=0.08, head_length=0.04, fc='green', ec='green', alpha=0.4)


        dist_ball_target = np.linalg.norm(self.ball - self.target)
        plt.gca().set_aspect("equal")
        plt.title(f"Step {self.step_count} | reward = {self.reward_value} m")
        plt.legend(loc="upper right", fontsize=8)
        
        plt.xlim(-3.2, 3.2)
        plt.ylim(-3.2, 3.2)
        plt.gca().set_aspect("equal")
        plt.pause(0.001)
