import time
import numpy as np
from envs.arm2dof_env import Arm2DoFEnv
from stable_baselines3 import PPO

# Charger le modèle entraîné
model_path = "models/test/ppo_arm2d_final.zip" 
model = PPO.load(model_path)

# Créer un environnement pour le test
env = Arm2DoFEnv(render_mode="human")

# Tester plusieurs épisodes 
num_tests = 5
max_steps = 100

for test_ep in range(num_tests):

    obs, _ = env.reset()
    total_reward = 0
    
    print(f"Test épisode {test_ep+1} : cible = {env.target}")

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Affichage
        env.render()
        time.sleep(0.02)

    if info["target_reached"]:
        print(f"  - Cible atteinte ! Récompense = {total_reward:.2f}")
    else:
        print(f"  - Échec après {max_steps} steps. Récompense = {total_reward:.2f}")

env.close()
