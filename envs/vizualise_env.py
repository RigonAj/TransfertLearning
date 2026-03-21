import time
import numpy as np

# IMPORT TES ENVIRONNEMENTS
from envs.arm2dof_env import Arm2DoFEnv
from envs.arm3dof_env import Arm3DoFEnv
from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof


def run_env(env_class, episodes=3):
    env = env_class(render_mode="human")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action = env.action_space.sample()  # action aléatoire
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            time.sleep(0.02)

    env.close()


if __name__ == "__main__":

    ENV1 = Arm3DoFEnv  
    ENV2 = Arm2DoFEnv
    ENV3 = PushBallEnv_2dof
    ENV4 = PushBallEnv_3dof
    
    run_env(ENV2)


