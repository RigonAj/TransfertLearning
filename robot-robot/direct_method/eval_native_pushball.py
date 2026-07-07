"""Taux de réussite natif d'une politique PPO push-ball 2DoF sur son propre env.

C'est le PLAFOND de tout transfert : un mapper parfait ne peut pas faire mieux.
Protocole identique au script de transfert : épisodes déterministes, max 300
pas, VecNormalize en mode inférence.

ATTENTION : préférer `ppo_pushball_final.zip` à `best_model.zip`. La sélection
"best" de l'EvalCallback repose sur le reward moyen de 20 épisodes (bruité, et
le reward shapé ne suit pas le taux de succès) et son vec_normalize.pkl est
écrasé par la sauvegarde finale. Mesuré sur ppo_pushball_2dof_4 (40 M) :
final = 99 %, best_model (snapshot 22 M) = 59 %.

Usage :
    cd robot-robot && python3 -m direct_method.eval_native_pushball
    python3 -m direct_method.eval_native_pushball --model-path ... --vecnorm-path ...
"""

import argparse

import numpy as np
import torch
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_2dof import PushBallEnv_2dof


def main():
    p = argparse.ArgumentParser(description="Évaluation native du PPO push-ball 2DoF.")
    p.add_argument("--model-path", default="data/models/ppo_pushball_2dof_4/ppo_pushball_final.zip")
    p.add_argument("--vecnorm-path", default="data/models/ppo_pushball_2dof_4/vec_normalize.pkl")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = PPO.load(args.model_path, device="cpu", custom_objects={
        "learning_rate": 3e-4, "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2})

    env = DummyVecEnv([lambda: Monitor(
        PushBallEnv_2dof(None, max_steps=args.max_steps))])
    vn = VecNormalize.load(args.vecnorm_path, env)
    vn.training = False
    vn.norm_reward = False
    vn.seed(args.seed)

    successes, steps_ok, dist_fail = 0, [], []
    for _ in range(args.episodes):
        obs = vn.reset()
        done, steps, info = False, 0, {}
        while not done:
            action, _ = model.predict(obs[0], deterministic=True)
            obs, _, dones, infos = vn.step(action.reshape(1, -1))
            steps += 1
            done = dones[0]
            info = infos[0]
        if info.get("target_reached", False):
            successes += 1
            steps_ok.append(steps)
        else:
            dist_fail.append(info.get("dist_ball_target", float("nan")))

    print("=" * 45)
    print(f"  Modèle                : {args.model_path}")
    print(f"  Épisodes              : {args.episodes} (seed {args.seed})")
    print(f"  Taux de réussite natif: {100 * successes / args.episodes:.1f}%")
    if steps_ok:
        print(f"  Steps moyens (succès) : {np.mean(steps_ok):.1f}")
    if dist_fail:
        print(f"  Distance moy (échec)  : {np.nanmean(dist_fail):.4f} m")
    print("=" * 45)


if __name__ == "__main__":
    main()
