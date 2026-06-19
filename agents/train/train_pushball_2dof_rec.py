if __name__ == "__main__":
    import os
    import argparse
    import numpy as np
    import torch
    import multiprocessing

    from torch import nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

    from envs.env_pushball_2dof_rec import PushBallEnv_2dof

    # ==============================
    # CLI arguments
    # ==============================
    parser = argparse.ArgumentParser(
        description="Entraînement PPO push-ball 2DoF avec récompense de "
                     "reconstruction via mappers (transfert 2DoF↔3DoF)."
    )
    # Coefficients linéaires et INDÉPENDANTS du reward de reconstruction :
    #   recon_reward = a * geometric_rec + b * effector_rec + c * velocity_rec
    # Chacun peut être réglé séparément pour étudier son impact isolément.
    parser.add_argument('--geometric_coef', '-a', dest='geometric_coef', type=float, default=0.5,
                         help="Coefficient a : poids de geometric_rec "
                              "(similarité de posture globale du bras, d_shape).")
    parser.add_argument('--effector_coef', '-b', dest='effector_coef', type=float, default=0.5,
                         help="Coefficient b : poids de effector_rec "
                              "(similarité de l'effecteur, d_ee).")
    parser.add_argument('--velocity_coef', '-c', dest='velocity_coef', type=float, default=0.5,
                         help="Coefficient c : poids de velocity_rec "
                              "(similarité des vélocités articulaires reconstruites).")

    # Chemins des mappers pré-entraînés (state/action, 2↔3 DoF)
    parser.add_argument('--state_mapper_2to3', type=str,
                         default='./models/mappers/state_mapper_2to3.pt',
                         help="Checkpoint du state mapper 2DoF → 3DoF (6→8).")
    parser.add_argument('--state_mapper_3to2', type=str,
                         default='./models/mappers/state_mapper_3to2.pt',
                         help="Checkpoint du state mapper 3DoF → 2DoF (8→6).")
    parser.add_argument('--action_mapper_2to3', type=str,
                         default='./models/mappers/action_mapper_2to3.pt',
                         help="Checkpoint de l'action mapper 2DoF → 3DoF "
                              "(état=8D, action_in=2D, action_out=3D).")
    parser.add_argument('--action_mapper_3to2', type=str,
                         default='./models/mappers/action_mapper_3to2.pt',
                         help="Checkpoint de l'action mapper 3DoF → 2DoF "
                              "(état=6D, action_in=3D, action_out=2D).")

    parser.add_argument('--run_id', type=int, default=1)
    args = parser.parse_args()

    # ==============================
    # CPU / Threads
    # ==============================
    torch.set_num_threads(16)

    # ==============================
    # Hyperparamètres
    # ==============================
    total_batch = 16384
    n_envs = 64
    TOTAL_TIMESTEPS = 150_000_000

    # ==============================
    # Directories
    # ==============================
    run_id = args.run_id
    run_name = f"ppo_pushball_2dof_rec_{run_id}"
    tensorboard_log_dir = f"./models/{run_name}/"
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

                geo = [i["geometric_rec"] for i in infos if "geometric_rec" in i]
                if geo:
                    self.logger.record("custom/geometric_rec", np.mean(geo))

                eff_rec = [i["effector_rec"] for i in infos if "effector_rec" in i]
                if eff_rec:
                    self.logger.record("custom/effector_rec", np.mean(eff_rec))

                vel_rec = [i["velocity_rec"] for i in infos if "velocity_rec" in i]
                if vel_rec:
                    self.logger.record("custom/velocity_rec", np.mean(vel_rec))

                recon = [i["recon_reward"] for i in infos if "recon_reward" in i]
                if recon:
                    self.logger.record("custom/recon_reward", np.mean(recon))
            return True

    # ==============================
    # Environment factory
    # ==============================
    def make_env(rank, seed=0):
        def _init():
            env = PushBallEnv_2dof(
                render_mode=None,
                state_mapper_2to3_path=args.state_mapper_2to3,
                state_mapper_3to2_path=args.state_mapper_3to2,
                action_mapper_2to3_path=args.action_mapper_2to3,
                action_mapper_3to2_path=args.action_mapper_3to2,
                geometric_coef=args.geometric_coef,
                effector_coef=args.effector_coef,
                velocity_coef=args.velocity_coef,
            )
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
        print(f"Reconstruction reward = a*geometric_rec + b*effector_rec + c*velocity_rec "
              f"| a={args.geometric_coef} b={args.effector_coef} c={args.velocity_coef}")

        # ==============================
        # Training environment (PARALLEL)
        # ==============================
        train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=0.99,
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
            gamma=0.99,
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

        n_steps = 2048
        batch_size = 1024

        model = PPO(
            "MlpPolicy",
            train_env,
            # --- collecte ---
            n_steps=n_steps,
            batch_size= batch_size,
            n_epochs=5,
            # --- optimisation ---
            learning_rate= 3e-4, 
            gamma=0.99,
            gae_lambda=0.95,
            # --- stabilisation ---
            clip_range=0.2,   
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.001, 
            vf_coef=0.5,
            max_grad_norm=0.5,
    

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
