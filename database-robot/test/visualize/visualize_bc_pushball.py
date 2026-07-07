"""Visualisation en direct d'une politique BC (ou PPO) sur push-ball 2DoF.

Affiche à chaque pas :
  - le bras 2DoF, l'effecteur, la balle (avec sa trajectoire), la cible et
    son rayon de succès ;
  - la flèche de vitesse d'effecteur commandée par la politique visualisée ;
  - avec --compare-expert : la flèche de ce que l'EXPERT aurait fait sur le
    même état — l'écart d'angle entre les deux est affiché dans la légende.
    C'est le diagnostic visuel du clonage : là où les flèches divergent,
    l'étudiant quitte la distribution experte (compounding errors).

Usage :
    cd database-robot

    # fenêtre interactive, modèle BC+DAgger (défaut)
    python3 test/visualize/visualize_bc_pushball.py --episodes 3

    # comparer visuellement les actions BC et expert
    python3 test/visualize/visualize_bc_pushball.py --episodes 3 --compare-expert

    # GIF par épisode, sans écran -> data/plots/bc_rollouts/
    MPLBACKEND=Agg python3 test/visualize/visualize_bc_pushball.py --episodes 3 --save
"""

import argparse
import sys
from pathlib import Path

# Rend `envs` importable quel que soit le répertoire de lancement
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.env_pushball_2dof import PushBallEnv_2dof

ROOT = Path(__file__).resolve().parents[2]
OMEGA_MAX = 2.0
ARROW_SCALE = 0.5   # s : longueur de flèche = vitesse (m/s) * échelle


def jacobian_2dof(t1, t2, l1, l2):
    s1, s12 = np.sin(t1), np.sin(t1 + t2)
    c1, c12 = np.cos(t1), np.cos(t1 + t2)
    return np.array([[-l1 * s1 - l2 * s12, -l2 * s12],
                     [ l1 * c1 + l2 * c12,  l2 * c12]])


def load_policy(model_dir):
    model_dir = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    model = PPO.load(model_dir / "best_model.zip", device="cpu", custom_objects={
        "learning_rate": 3e-4, "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2})
    return model, model_dir


def angle_between(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    c = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def main():
    p = argparse.ArgumentParser(
        description="Visualiser une politique BC/PPO sur push-ball 2DoF (avec légende).")
    p.add_argument("--model-dir", default="data/models/bc_pushball_dagger",
                   help="dossier contenant best_model.zip + vec_normalize.pkl")
    p.add_argument("--expert-dir", default="database/models/ppo_pushball_2dof_1",
                   help="politique experte pour --compare-expert")
    p.add_argument("--compare-expert", action="store_true",
                   help="superposer la flèche d'action de l'expert sur les mêmes états")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.03)
    p.add_argument("--save", action="store_true",
                   help="écrire un GIF par épisode dans data/plots/bc_rollouts/ "
                        "(utiliser MPLBACKEND=Agg sans écran)")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    interactive = matplotlib.get_backend().lower() != "agg" and not args.save

    model, model_dir = load_policy(args.model_dir)
    expert = None
    if args.compare_expert:
        expert, _ = load_policy(args.expert_dir)

    env = DummyVecEnv([lambda: PushBallEnv_2dof(None, max_steps=args.max_steps)])
    vn = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), env)
    vn.training = False
    vn.norm_reward = False
    vn.seed(args.seed)
    inner = env.envs[0].unwrapped

    gif_dir = ROOT / "data" / "plots" / "bc_rollouts"
    model_name = model_dir.name

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    if interactive:
        plt.ion()
        plt.show()

    for ep in range(args.episodes):
        obs = vn.reset()
        frames, ball_trace = [], []
        done, step, info = False, 0, {}

        while not done:
            o = obs[0].astype(np.float32)
            action, _ = model.predict(o, deterministic=True)

            t1, t2 = inner.theta1, inner.theta2
            J = jacobian_2dof(t1, t2, inner.l1, inner.l2)
            v_bc = J @ (action * OMEGA_MAX)
            v_exp, div = None, None
            if expert is not None:
                a_exp, _ = expert.predict(o, deterministic=True)
                v_exp = J @ (a_exp * OMEGA_MAX)
                div = angle_between(v_bc, v_exp)

            obs, _, dones, infos = vn.step(action.reshape(1, -1))
            done = dones[0]
            info = infos[0]
            step += 1
            ball_trace.append(inner.ball.copy())

            # ── dessin ────────────────────────────────────────────────
            ax.cla()
            a = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(a) * inner.max_reach, np.sin(a) * inner.max_reach,
                    color="#777777", linestyle=":", lw=1.0, alpha=0.6,
                    label="Espace de travail (r = 3 m)")

            j1 = np.array([inner.l1 * np.cos(inner.theta1),
                           inner.l1 * np.sin(inner.theta1)])
            eff = j1 + np.array([inner.l2 * np.cos(inner.theta1 + inner.theta2),
                                 inner.l2 * np.sin(inner.theta1 + inner.theta2)])
            ax.plot([0, j1[0], eff[0]], [0, j1[1], eff[1]], "-o",
                    color="#1f77b4", lw=4, markersize=5, label="Bras 2DoF")
            ax.add_patch(plt.Circle(eff, inner.eff_radius, color="#1f77b4",
                                    alpha=0.9, label="Effecteur"))
            ax.add_patch(plt.Circle(inner.ball, inner.ball_radius,
                                    color="#E05A3A", alpha=0.85, label="Balle"))
            bt = np.array(ball_trace)
            ax.plot(bt[:, 0], bt[:, 1], "--", color="#E05A3A", lw=1, alpha=0.5,
                    label="Trajectoire balle")
            ax.add_patch(plt.Circle(inner.target, inner.epsilon, color="green",
                                    fill=False, linestyle="--", lw=1.8,
                                    label=f"Cible (succès < {inner.epsilon:.2f} m)"))
            ax.plot(*inner.target, "+", color="green", markersize=10, markeredgewidth=2)

            if np.linalg.norm(v_bc) > 1e-3:
                ax.annotate("", xy=eff + v_bc * ARROW_SCALE, xytext=eff,
                            arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2.2))
            ax.plot([], [], color="#1f77b4", lw=2.2,
                    label=f"v_eff politique ({model_name}, ‖v‖={np.linalg.norm(v_bc):.2f} m/s)")
            if v_exp is not None:
                if np.linalg.norm(v_exp) > 1e-3:
                    ax.annotate("", xy=eff + v_exp * ARROW_SCALE, xytext=eff,
                                arrowprops=dict(arrowstyle="-|>", color="#9B5DE5",
                                                lw=2.2, alpha=0.9))
                ax.plot([], [], color="#9B5DE5", lw=2.2,
                        label=f"v_eff expert (écart = {div:.0f}°)")

            title = (f"Push-ball 2DoF — politique : {model_name}\n"
                     f"épisode {ep + 1}/{args.episodes} — pas {step} | "
                     f"d(balle, cible) = {info['dist_ball_target']:.3f} m")
            if done:
                title += ("  ->  SUCCÈS" if info.get("target_reached")
                          else "  ->  ÉCHEC (balle sortie)"
                          if float(np.linalg.norm(inner.ball)) > inner.max_reach * 1.05
                          else "  ->  ÉCHEC (temps écoulé)")

            ax.set_xlim(-3.4, 3.4)
            ax.set_ylim(-3.4, 3.4)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(title, fontsize=10)
            ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

            if interactive:
                plt.pause(args.delay if not done else 1.5)
            if args.save:
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

        result = "succes" if info.get("target_reached") else "echec"
        print(f"épisode {ep + 1}: {result} en {step} pas "
              f"(d finale = {info['dist_ball_target']:.3f} m)")

        if args.save and frames:
            from PIL import Image
            gif_dir.mkdir(parents=True, exist_ok=True)
            path = gif_dir / f"bc_{model_name}_ep{ep + 1:02d}_{result}.gif"
            imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE)
                    for f in frames]
            imgs[0].save(path, save_all=True, append_images=imgs[1:],
                         duration=[1000 // args.fps] * (len(imgs) - 1) + [1500],
                         loop=0)
            print(f"  GIF → {path}")

    if interactive:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
