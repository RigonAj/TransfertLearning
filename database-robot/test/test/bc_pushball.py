import argparse
import sys

import numpy as np
from pathlib import Path

# Rend `envs` importable quel que soit le répertoire de lancement (comme train_bc.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.env_pushball_2dof import PushBallEnv_2dof

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "data" / "models").exists())
MODEL_ROOT = ROOT / "data" / "models"

# run_id passé en argument au lancement du script pour choisir le modèle à tester
parser = argparse.ArgumentParser(description="Test BC model on push-ball 2-dof environment")
parser.add_argument("--run_id", type=int, default=1, help="Run ID of the BC model to test")
parser.add_argument("--model-dir", default=None,
                    help="dossier du modèle (ex: data/models/bc_pushball_dagger) ; "
                         "prioritaire sur --run_id")
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--max-steps", type=int, default=150)
args = parser.parse_args()

# --- Configuration ---
if args.model_dir is not None:
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
else:
    model_dir = MODEL_ROOT / f"bc_pushball_{args.run_id}"
MODEL_PATH  = model_dir / "best_model.zip"
VECNORM_PATH = model_dir / "vec_normalize.pkl"
NUM_EPISODES = args.episodes
MAX_STEPS    = args.max_steps

# --- Charger le modèle ---
model = PPO.load(MODEL_PATH, custom_objects={"learning_rate": 0.0003, "lr_schedule": lambda _: 0.0003, "clip_range": lambda _: 0.2})

# --- Créer l'environnement AVEC VecNormalize (indispensable !) ---
def make_env():
    # MAX_STEPS doit être transmis à l'env (avant : il restait au défaut 100)
    return Monitor(PushBallEnv_2dof(render_mode=None, max_steps=MAX_STEPS))

env = DummyVecEnv([make_env])
env = VecNormalize.load(VECNORM_PATH, env)
env.training = False      # ne pas mettre à jour les stats
env.norm_reward = False   # on veut la reward brute pour les stats

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
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
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
