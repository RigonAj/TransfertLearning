import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.env_pushball_2dof import PushBallEnv_2dof

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "data" / "models").exists())
MODEL_ROOT = ROOT / "data" / "models"

parser = argparse.ArgumentParser(description="Test BC model on push-ball 2-dof environment")
parser.add_argument("--run_id", type=int, default=1, help="Run ID of the BC model to test")
args = parser.parse_args()

run_id = args.run_id
model_dir = MODEL_ROOT / f"bc_pushball_{run_id}"
VECNORM_PATH = model_dir / "vec_normalize.pkl"

# Liste des fichiers checkpoint
checkpoint_files = sorted(model_dir.glob("checkpoint_epoch_*.zip"),
                          key=lambda x: int(x.stem.split("_")[-1]))

if not checkpoint_files:
    print(f"Aucun checkpoint trouvé dans {model_dir}")
    exit()

NUM_EPISODES = 1000

# Fonction de test pour un modèle donné
def test_model(model, vec_norm_env, num_episodes):
    successes = 0
    steps_on_success = []
    final_dist_on_failure = []

    for ep in range(num_episodes):
        obs = vec_norm_env.reset()
        done = False
        step_count = 0
        info_last = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_norm_env.step(action)
            step_count += 1
            done = dones[0]
            info_last = infos[0]

        if info_last.get("target_reached", False):
            successes += 1
            steps_on_success.append(step_count)
        else:
            final_dist_on_failure.append(info_last.get("dist_ball_target", float("nan")))

    success_rate = 100 * successes / num_episodes
    return successes, success_rate, steps_on_success, final_dist_on_failure

# Créer l'environnement de base (avec VecNormalize) – partagé pour tous les checkpoints
def make_env():
    return Monitor(PushBallEnv_2dof(render_mode=None))

base_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load(VECNORM_PATH, base_env)
vec_env.training = False      # ne pas mettre à jour les stats
vec_env.norm_reward = False   # reward brute pour les stats

# Tester chaque checkpoint
for checkpoint_file in checkpoint_files:
    checkpoint_epoch = int(checkpoint_file.stem.split("_")[-1])
    checkpoint_name = checkpoint_file.stem  # ex: checkpoint_epoch_42

    print(f"\n--- Test du checkpoint : {checkpoint_name} (epoch {checkpoint_epoch}) ---")

    # Charger le modèle
    model = PPO.load(checkpoint_file,
                     custom_objects={
                         "learning_rate": 0.0003,
                         "lr_schedule": lambda _: 0.0003,
                         "clip_range": lambda _: 0.2
                     })

    # Exécuter les 500 épisodes
    successes, success_rate, steps_on_success, final_dist_on_failure = test_model(
        model, vec_env, NUM_EPISODES
    )

    # Affichage compact demandé : nom du checkpoint + pourcentage
    #print(f"Checkpoint : {checkpoint_name}")
    print(f"Taux de réussite : {success_rate:.1f}%")
    '''
    # Détails supplémentaires (optionnels)
    print(f"  Épisodes réussis : {successes}/{NUM_EPISODES}")
    if steps_on_success:
        print(f"  Steps moyens (succès) : {np.mean(steps_on_success):.1f}")
    if final_dist_on_failure:
        print(f"  Distance moyenne (échec) : {np.mean(final_dist_on_failure):.4f} m")
    print("-" * 45)
    '''

# Fermeture propre
vec_env.close()