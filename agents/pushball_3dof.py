import matplotlib
matplotlib.use("TkAgg")  # ou "Qt5Agg" se tiver PyQt5 instalado
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.env_pushball_3dof import PushBallEnv_3dof
from direct_method.mapper_models import StateMapperMLP, ActionMapperMLP


# --- Configuration ---
run_id = 1
MODEL_PATH   = f"models/ppo_pushball_2dof_{run_id}_theta_constrained/best_model.zip"
VECNORM_PATH = f"models/ppo_pushball_3dof_{run_id}_theta_constrained/vec_normalize.pkl"

NUM_EPISODES = 100
MAX_STEPS    = 300

# --- Charger le modèle ---
model = PPO.load(MODEL_PATH, custom_objects={"learning_rate": 0.0003, "lr_schedule": lambda _: 0.0003, "clip_range": lambda _: 0.2})

# state and action mappers
R1_STATE_DIM = 4
R2_STATE_DIM = 6
R1_ACTION_DIM = 2
R2_ACTION_DIM = 3
HIDDEN_DIM = 256

path_state_mapper_r2_to_r1 = f"runs/run_01/models/state_mapper_r2_to_r1.pt"
path_action_mapper_r1_to_r2 = f"runs/run_02/models/action_mapper_r1_to_r2.pt"

state_mapper_r2_to_r1 = StateMapperMLP(R2_STATE_DIM, R1_STATE_DIM, HIDDEN_DIM)
state_mapper_r2_to_r1.load_state_dict(torch.load(path_state_mapper_r2_to_r1, map_location="cpu"))
state_mapper_r2_to_r1.eval()

action_mapper_r1_to_r2 = ActionMapperMLP(R1_ACTION_DIM, R2_ACTION_DIM, HIDDEN_DIM)
action_mapper_r1_to_r2.load_state_dict(torch.load(path_action_mapper_r1_to_r2, map_location="cpu"))
action_mapper_r1_to_r2.eval()

def relative_to_absolute(q_rel):
    q_abs = np.cumsum(q_rel)
    return (q_abs + np.pi) % (2 * np.pi) - np.pi

def absolute_to_relative(q_abs):

    q_abs = np.array(q_abs)

    q_rel = np.empty_like(q_abs)
    q_rel[0] = q_abs[0]

    q_rel[1:] = np.diff(q_abs)

    return (q_rel + np.pi) % (2*np.pi) - np.pi


# --- Créer l'environnement AVEC VecNormalize (indispensable !) ---
def make_env():
    return Monitor(PushBallEnv_3dof(render_mode=None,max_steps=MAX_STEPS))

env = DummyVecEnv([make_env])
env = VecNormalize.load(VECNORM_PATH, env)
env.training = False      # ne pas mettre à jour les stats
env.norm_reward = False   # on veut la reward brute pour les stats
env.norm_obs = False


# --- Statistiques ---
successes = 0
steps_on_success = []
final_dist_on_failure = []

print(f"Lancement de {NUM_EPISODES} épisodes de test...\n")

for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    step_count = 0
    info_last = {}
    

    while not done:
        # get robot state
        #print(obs)
        state_r2 = obs[0, :R2_STATE_DIM].copy()
        #print(state_r2)
        state_r2_t = torch.from_numpy(state_r2).float().unsqueeze(0)
        with torch.no_grad():
            state_r1 = state_mapper_r2_to_r1(state_r2_t).squeeze(0).numpy()
            #print(state_r1)
        new_obs = np.concatenate((state_r1, obs[0, R2_STATE_DIM:]))
        # get action r1
        action_r1, _ = model.predict(new_obs, deterministic=True)
        action_r1_t = torch.from_numpy(action_r1.copy()).float().unsqueeze(0)
        #print(action_r1_t)
        with torch.no_grad():
            action_r2 = action_mapper_r1_to_r2(action_r1_t.clamp(-2, 2)*0.5).squeeze(0).numpy()

        # compute action r2
        action_r2 = action_r2.reshape(1, -1)
        #print(action_r2)
        obs, reward, dones, infos = env.step(action_r2)
        #env.render()
        #print("=" * 45)
        step_count += 1
        done = dones[0]
        info_last = infos[0]
        

    if info_last.get("target_reached", False):
        successes += 1
        steps_on_success.append(step_count)
    else:
        final_dist_on_failure.append(info_last.get("dist_ball_target", float("nan")))

# --- Résultats ---
success_rate = 100 * successes / NUM_EPISODES
print("=" * 45)
print(f"  Épisodes testés       : {NUM_EPISODES}")
print(f"  Réussites             : {successes}")
print(f"  Taux de réussite      : {success_rate:.1f}%")
print("-" * 45)
if steps_on_success:
    print(f"  Steps moyens (succès) : {np.mean(steps_on_success):.1f}")
    print(f"  Steps min / max       : {np.min(steps_on_success)} / {np.max(steps_on_success)}")
if final_dist_on_failure:
    print(f"  Distance moy (échec)  : {np.mean(final_dist_on_failure):.4f} m")
    print(f"  Distance min / max    : {np.min(final_dist_on_failure):.4f} / {np.max(final_dist_on_failure):.4f} m")
print("=" * 45)

env.close()
