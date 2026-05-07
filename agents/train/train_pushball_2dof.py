if __name__ == "__main__":
    import os
    import numpy as np
    import torch
    import multiprocessing

    from torch import nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

    from envs.env_pushball_2dof import PushBallEnv_2dof
    
    # ==============================
    # CPU / Threads
    # ==============================
    torch.set_num_threads(16)

    # ==============================
    # Hyperparamètres
    # ==============================
    total_batch = 16384
    n_envs = 64
    TOTAL_TIMESTEPS = 50_000_000

    # ==============================
    # Directories
    # ==============================
    run_id = 1
    run_name = f"ppo_pushball_2dof_{run_id}"
    tensorboard_log_dir = f"./logs/{run_name}/"
    model_dir = f"./models/{run_name}/"

    # ==============================
    # Schedules
    # ==============================
    def linear_schedule(initial_value):
        def func(progress_remaining):
            return progress_remaining * initial_value
        return func

    # ==============================
    # Sync VecNormalize stats
    # ==============================
    def sync_envs_normalization(train_env, eval_env):
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms

    class SyncNormEvalCallback(EvalCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_mean_reward = -np.inf

        def _on_step(self) -> bool:
            sync_envs_normalization(self.training_env, self.eval_env)

            result = super()._on_step()

            if self.last_mean_reward > self.best_mean_reward:
                self.best_mean_reward = self.last_mean_reward
                vec_path = os.path.join(self.best_model_save_path, "vec_normalize.pkl")
                self.training_env.save(vec_path)

            return result

    class CustomTensorboardCallback(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            if len(infos) > 0:
                alignments = [i["alignment"] for i in infos if "alignment" in i]
                if alignments:
                    self.logger.record("custom/alignment", np.mean(alignments))

                d_bt = [i["dist_ball_target"] for i in infos if "dist_ball_target" in i]
                if d_bt:
                    self.logger.record("custom/dist_ball_target", np.mean(d_bt))

                d_eb = [i["dist_eff_ball"] for i in infos if "dist_eff_ball" in i]
                if d_eb:
                    self.logger.record("custom/dist_eff_ball", np.mean(d_eb))

                p_bt = [i["progress_ball_target"] for i in infos if "progress_ball_target" in i]
                if p_bt:
                    self.logger.record("custom/progress_ball_target", np.mean(p_bt))
            return True

    # ==============================
    # Environment factory
    # ==============================
    def make_env(rank, seed=0):
        def _init():
            env = PushBallEnv_2dof(render_mode=None)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    # Vérification : seul le processus principal exécute l'entraînement
    if multiprocessing.current_process().name == "MainProcess":
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(tensorboard_log_dir, exist_ok=True)

        print(f"Logs  : {tensorboard_log_dir}")
        print(f"Models: {model_dir}")

        # ==============================
        # Training environment (PARALLEL)
        # ==============================
        train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=0.9,
        )

        # ==============================
        # Evaluation environment (SINGLE ENV)
        # ==============================
        eval_env = DummyVecEnv([make_env(0)])
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            gamma=0.9,
            training=False,
        )

        # ==============================
        # Evaluation callback
        # ==============================
        eval_callback = SyncNormEvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=tensorboard_log_dir,
            eval_freq=5_000,
            n_eval_episodes=20,
            deterministic=True,
            render=False,
            verbose=1,
        )

        # ==============================
        # Policy — 2 × 256, Tanh
        # ==============================
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh,
        )

        # ==============================
        # PPO
        # ==============================

        n_steps = total_batch // n_envs
        batch_size = 512

        assert (n_steps * n_envs) % batch_size == 0

        model = PPO(
            "MlpPolicy",
            train_env,
            # --- collecte ---
            n_steps=n_steps,
            batch_size=128,
            n_epochs=5,
            # --- optimisation ---
            learning_rate=linear_schedule(5e-5), 
            gamma=0.9,
            gae_lambda=0.9,
            # --- stabilisation ---
            clip_range=0.1,   
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.00, 
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.02, 

            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log_dir,
            verbose=1,
        )

        # ==============================
        # Training
        # ==============================
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_callback, CustomTensorboardCallback()],
            progress_bar=True,
        )

        # ==============================
        # Save final
        # ==============================
        model_path = os.path.join(model_dir, "ppo_pushball_final")
        vec_path = os.path.join(model_dir, "vec_normalize.pkl")

        model.save(model_path)
        train_env.save(vec_path)

        print("Training finished.")
        print(f"Model  : {model_path}")
        print(f"VecNorm: {vec_path}")
