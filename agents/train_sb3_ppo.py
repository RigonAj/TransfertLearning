import os
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
def sync_envs_normalization(train_env, eval_env):
    """Copy VecNormalize running stats from train_env to eval_env."""
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
from envs.arm2dof_env import Arm2DoFEnv

# ==============================
# Directories
# ==============================
run_name        = "ppo_reach_v2"
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
    return Monitor(Arm2DoFEnv(render_mode=None))


# ==============================
# Training environment
# ==============================
train_env = DummyVecEnv([make_env])
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,   # ON: reward has mixed scales (-d, bonus, penalty)
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99,
)

# ==============================
# Evaluation environment
# ==============================
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,  # off for eval so mean_reward is interpretable
    training=False,
    clip_obs=10.0,
)


# ==============================
# Evaluation callback with proper VecNormalize sync
# ==============================
class SyncedEvalCallback(EvalCallback):
    """Syncs VecNormalize stats before each eval so obs normalisation is correct."""
    def _on_step(self) -> bool:
        sync_envs_normalization(self.training_env, self.eval_env)
        return super()._on_step()


eval_callback = SyncedEvalCallback(
    eval_env,
    best_model_save_path=model_dir,
    log_path=model_dir,
    eval_freq=5000,
    n_eval_episodes=20,
    deterministic=True,
    render=False,
)

# ==============================
# Policy
# ==============================
policy_kwargs = dict(
    activation_fn=nn.Tanh,
    net_arch=[256, 256],
)

# ==============================
# PPO model
# ==============================
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.001,     # fixed, low — no schedule (entropy was exploding)
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.02,     # early-stop updates when KL > 0.02 (was reaching 0.05+)
    verbose=1,
    tensorboard_log=tensorboard_log_dir,
    policy_kwargs=policy_kwargs,
)

# ==============================
# Training
# ==============================
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback,
)

# ==============================
# Save
# ==============================
model_path = os.path.join(model_dir, "ppo_arm2d_final")
vec_path   = os.path.join(model_dir, "vec_normalize.pkl")

model.save(model_path)
train_env.save(vec_path)

print("Training finished.")
print(f"Model : {model_path}")
print(f"VecNorm: {vec_path}")
