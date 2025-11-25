import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from envs.arm2d_env import Arm2DEnv

#creer environnements
train_env = Arm2DEnv(render_mode=None)
eval_env = Monitor(Arm2DEnv(render_mode=None))  # Monitor pour éviter le warning


#dossier sauvegarde
log_dir = "models/"
os.makedirs(log_dir, exist_ok=True)

#callback evaluation
eval_callback = EvalCallback(
	eval_env,
	best_model_save_path=log_dir,
	log_path=log_dir,
	eval_freq=2000,   # évaluer tous les 2000 steps
	deterministic=True,
	render=False 
)

#agent PPO #########################################
model = PPO(
	"MlpPolicy",
	train_env,
	n_steps=128, 
	batch_size=32,
	verbose=1,
	learning_rate=3e-4)

#entrainement

model.learn(
	total_timesteps=20000,  #################################
	callback=eval_callback
    )
#sauvegarde
model.save(os.path.join(log_dir, "ppo_arm2d"))
print(log_dir)


