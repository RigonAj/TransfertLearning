"""
Entraînement PPO — PushBallEnv_2dof

"""

import os
import numpy as np
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
#from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_2dof_ref import PushBallEnv_2dof


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
    def _on_step(self) -> bool:
        sync_envs_normalization(self.training_env, self.eval_env)
        return super()._on_step()


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
# Hyperparamètres
# ==============================
TOTAL_TIMESTEPS = 3_000_000 

# ==============================
# Directories
# ==============================
run_id              = 3
run_name            = f"ppo_pushball_2dof_{run_id}"
tensorboard_log_dir = f"./logs/{run_name}/"
model_dir           = f"./models/{run_name}/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

print(f"Logs  : {tensorboard_log_dir}")
print(f"Models: {model_dir}")


# ==============================
# Environment factory
# ==============================
def make_env():
    return Monitor(PushBallEnv_2dof(render_mode=None))


# ==============================
# Training environment
# ==============================
train_env = DummyVecEnv([make_env])
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    gamma=0.9,
)

# ==============================
# Evaluation environment
# ==============================
eval_env = DummyVecEnv([make_env])
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
# Policy — 2 × 256, Tanh (rl-zoo3 Pusher)
# ==============================
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=nn.Tanh,
)

# ==============================
# PPO — rl-baselines3-zoo Pusher-v4
# ==============================
model = PPO(
    "MlpPolicy",
    train_env,
    # --- collecte ---
    n_steps=1024,
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
    ent_coef=0.001, 
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
vec_path   = os.path.join(model_dir, "vec_normalize.pkl")

model.save(model_path)
train_env.save(vec_path)

print("Training finished.")
print(f"Model  : {model_path}")
print(f"VecNorm: {vec_path}")
