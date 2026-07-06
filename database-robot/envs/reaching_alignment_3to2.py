import numpy as np

from envs.env_reaching_2dof import ReachingEnv_2dof


class ReachingAlignment3to2Env(ReachingEnv_2dof):

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

        # ==========================
        # Expert (3DoF)
        # ==========================
        self.expert_env = expert_env
        self.expert_policy = expert_policy

        self.lambda_pos = lambda_pos
        self.lambda_vel = lambda_vel

        self.deterministic_expert = deterministic_expert

        # tracking
        self.last_pos_error = 0.0
        self.last_vel_error = 0.0

    # ==========================================================
    # RESET (synchronisation cible)
    # ==========================================================
    def reset(self, seed=None, options=None):

        obs2, info = super().reset(seed=seed, options=options)

        # reset expert 3DoF
        self.expert_env.reset(seed=seed, options=options)

        # synchronisation cible
        self.expert_env.target = self.target.copy()

        # état initial alignement
        ee2 = self.get_end_effector_pos()
        ee3 = self.expert_env.get_end_effector_pos()

        vel2 = self.get_end_effector_vel()
        vel3 = self.expert_env.get_end_effector_vel()

        self.last_pos_error = np.linalg.norm(ee2 - ee3)
        self.last_vel_error = np.linalg.norm(vel2 - vel3)

        return obs2, info

    # ==========================================================
    # STEP
    # ==========================================================
    def step(self, action):

        # ==========================
        # Expert 3DoF
        # ==========================
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

        # ==========================
        # Student 2DoF (env parent)
        # ==========================
        obs, reward, terminated, truncated, info = super().step(action)

        # ==========================
        # EE states
        # ==========================
        ee2 = self.get_end_effector_pos()
        ee3 = self.expert_env.get_end_effector_pos()

        vel2 = self.get_end_effector_vel()
        vel3 = self.expert_env.get_end_effector_vel()

        # ==========================
        # ALIGNEMENT (RELATIF CIBLE)
        # ==========================
        target = self.target

        # positions relatives
        rel2 = ee2 - target
        rel3 = ee3 - target

        pos_error = np.linalg.norm(rel2 - rel3)

        # vitesses
        vel_error = np.linalg.norm(vel2 - vel3)

        alignment_reward = (
            - self.lambda_pos * pos_error
            - self.lambda_vel * vel_error
        )

        reward += alignment_reward

        # ==========================
        # infos debug
        # ==========================
        self.last_pos_error = pos_error
        self.last_vel_error = vel_error

        info["alignment_reward"] = alignment_reward
        info["pos_error"] = pos_error
        info["vel_error"] = vel_error

        info["expert_done"] = expert_terminated or expert_truncated

        info["expert_distance"] = np.linalg.norm(
            ee3 - self.expert_env.target
        )

        info["student_distance"] = np.linalg.norm(
            ee2 - self.target
        )

        return obs, reward, terminated, truncated, info