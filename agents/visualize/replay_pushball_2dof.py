"""
replay_pushball_2dof.py
Rejoue les épisodes sauvegardés par runs_pushball_2dof.py.

Usage :
  python replay_pushball_2dof.py              # rejoue les succès (défaut)
  python replay_pushball_2dof.py --success
  python replay_pushball_2dof.py --fail
  python replay_pushball_2dof.py --fail --ep 3
  python replay_pushball_2dof.py --success --delay 0.05
"""

import argparse
import glob
import os
import time

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Replay push-ball 2-dof episodes")
group = parser.add_mutually_exclusive_group()
group.add_argument("--success", action="store_true", default=False)
group.add_argument("--fail",    action="store_true", default=False)
parser.add_argument("--ep",    type=int, default=None)
parser.add_argument("--delay", type=float, default=0.05)
args = parser.parse_args()

mode     = "fail" if args.fail else "success"
data_dir = f"runs/pushball_2dof/{mode}"

# ── Chargement ────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(data_dir, "ep_*.npz")))
if not files:
    raise FileNotFoundError(
        f"Aucun fichier dans '{data_dir}'. Lancez runs_pushball_2dof.py d'abord."
    )
if args.ep is not None:
    target_name = f"ep_{args.ep:03d}.npz"
    files = [f for f in files if os.path.basename(f) == target_name]
    if not files:
        raise FileNotFoundError(f"Épisode {args.ep:03d} introuvable dans {data_dir}.")

print(f"Replay [{mode.upper()}] — {len(files)} épisode(s) — push-ball 2-DOF\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fk(t1, t2, l1, l2):
    x = l1*np.cos(t1) + l2*np.cos(t1+t2)
    y = l1*np.sin(t1) + l2*np.sin(t1+t2)
    return np.array([x, y])

# ── Replay ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
plt.ion()

for file_idx, fpath in enumerate(files):
    ep_name = os.path.basename(fpath)
    d = np.load(fpath, allow_pickle=True)

    target         = d["target"]
    thetas         = d["thetas"]          # (T, 2)
    balls          = d["balls"]           # (T, 2)
    rewards        = d["rewards"]
    dists_ball_tgt = d["dists_ball_tgt"]
    l1             = float(d["l1"])
    l2             = float(d["l2"])
    epsilon        = float(d["epsilon"])
    eff_radius     = float(d["eff_radius"])
    ball_radius    = float(d["ball_radius"])
    max_reach      = float(d["max_reach"])
    total_r        = float(d["total_reward"])
    success        = bool(d["success"])

    label = "✓ Succès" if success else "✗ Échec"
    print(f"[{file_idx+1}/{len(files)}]  {ep_name}  |  {label}  |  "
          f"steps={len(thetas)}  reward={total_r:+.1f}  "
          f"dist_finale={dists_ball_tgt[-1]:.4f} m")

    for step_i, (t1, t2) in enumerate(thetas):
        ax.cla()

        j1  = np.array([l1*np.cos(t1), l1*np.sin(t1)])
        eff = fk(t1, t2, l1, l2)
        ball          = balls[step_i]
        dist_ball_tgt = dists_ball_tgt[step_i]
        ball_out      = float(np.linalg.norm(ball)) > max_reach * 1.05

        # Bras
        ax.plot([0, j1[0]], [0, j1[1]], 'r-', lw=4, label='Link 1')
        ax.plot([j1[0], eff[0]], [j1[1], eff[1]], 'b-', lw=4, label='Link 2')
        ax.add_patch(plt.Circle(eff, eff_radius * 2, color='red', alpha=0.6))

        # Balle
        ball_color = 'red' if ball_out else 'dodgerblue'
        ax.add_patch(plt.Circle(ball, ball_radius * 1.5, color=ball_color, alpha=0.6))

        # Cible
        ax.plot(target[0], target[1], 'o', markersize=18, label='Target', color='orange')
        ax.add_patch(plt.Circle(target, epsilon,
                                color='green', fill=False, linestyle='--', lw=1.5))

        # Flèche balle → cible
        vec = target - ball
        if np.linalg.norm(vec) > 1e-3:
            v = vec / np.linalg.norm(vec) * 0.3
            ax.arrow(ball[0], ball[1], v[0], v[1],
                     head_width=0.08, head_length=0.04,
                     fc='green', ec='green', alpha=0.4)

        # Trajectoire balle
        ax.plot(balls[:step_i+1, 0], balls[:step_i+1, 1],
                '--', color='dodgerblue', lw=1, alpha=0.4, label='Traj. balle')

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_title(
            f"{ep_name}  [{mode}]  step {step_i+1}/{len(thetas)}\n"
            f"d(ball,tgt)={dist_ball_tgt:.3f} m"
            + ("  ⚠️ OUT" if ball_out else "")
            + f"  |  reward_cumulé={np.sum(rewards[:step_i+1]):+.1f}"
        )
        ax.legend(loc="upper right", fontsize=8)
        plt.pause(args.delay)

    time.sleep(0.5)

plt.ioff()
plt.show()
print("\nReplay terminé.")
