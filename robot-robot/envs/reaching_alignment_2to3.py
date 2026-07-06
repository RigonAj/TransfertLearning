import numpy as np

from envs.env_reaching_3dof import ReachingEnv_3dof


class ReachingAlignment2to3Env(ReachingEnv_3dof):

    def __init__(
        self,
        expert_env,
        expert_policy,
        lambda_pos=0.5,
        lambda_vel=0.1,
        deterministic_expert=True,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.expert_env = expert_env
        self.expert_policy = expert_policy

        self.lambda_pos = lambda_pos
        self.lambda_vel = lambda_vel

        self.deterministic_expert = deterministic_expert

        self.last_pos_error = 0.0
        self.last_vel_error = 0.0

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self, seed=None, options=None):

        obs3, info = super().reset(
            seed=seed,
            options=options
        )

        self.expert_env.reset(
            seed=seed,
            options=options
        )

        # même cible pour les deux robots
        self.expert_env.target = self.target.copy()

        eff2 = self.expert_env.get_end_effector_pos()
        eff3 = self.get_end_effector_pos()

        vel2 = self.expert_env.get_end_effector_vel()
        vel3 = self.get_end_effector_vel()

        self.last_pos_error = np.linalg.norm(
            eff2 - eff3
        )

        self.last_vel_error = np.linalg.norm(
            vel2 - vel3
        )

        return obs3, info

    # ==========================================================
    # STEP
    # ==========================================================

    def step(self, action):

        # ----------------------------------------
        # Expert 2DoF
        # ----------------------------------------

        expert_obs = self.expert_env._get_obs()

        expert_action, _ = self.expert_policy.predict(
            expert_obs,
            deterministic=self.deterministic_expert
        )

        (
            _,
            _,
            expert_terminated,
            expert_truncated,
            _
        ) = self.expert_env.step(expert_action)

        # ----------------------------------------
        # Agent 3DoF
        # ----------------------------------------

        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = super().step(action)

        # ----------------------------------------
        # Alignement espace tâche
        # ----------------------------------------

        ee2 = self.expert_env.get_end_effector_pos()
        ee3 = self.get_end_effector_pos()

        vel2 = self.expert_env.get_end_effector_vel()
        vel3 = self.get_end_effector_vel()

        pos_error = np.linalg.norm( (ee2-self.target) - (ee3-self.target) )

        vel_error = np.linalg.norm( (vel2-self.target) - (vel3-self.target) )

        alignment_reward = (
            - self.lambda_pos * pos_error
            - self.lambda_vel * vel_error
        )

        reward += alignment_reward

        self.last_pos_error = pos_error
        self.last_vel_error = vel_error

        # ----------------------------------------
        # Debug infos
        # ----------------------------------------

        info["alignment_reward"] = alignment_reward

        info["pos_error"] = pos_error
        info["vel_error"] = vel_error

        info["expert_done"] = (
            expert_terminated
            or
            expert_truncated
        )

        info["expert_distance"] = np.linalg.norm(
            ee2 - self.expert_env.target
        )

        info["student_distance"] = np.linalg.norm(
            ee3 - self.target
        )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )