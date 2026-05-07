"""
replay_reach_2dof.py
Rejoue les épisodes sauvegardés par runs_reach_2dof.py.

Usage :
  python replay_reach_2dof.py              # rejoue les succès (défaut)
  python replay_reach_2dof.py --success
  python replay_reach_2dof.py --fail
  python replay_reach_2dof.py --fail --ep 3      # rejoue uniquement l'épisode 003
  python replay_reach_2dof.py --success --delay 0.05
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
parser = argparse.ArgumentParser(description="Replay reaching-2dof episodes")
group = parser.add_mutually_exclusive_group()
group.add_argument("--success", action="store_true", default=False,
                   help="Rejouer les épisodes réussis (défaut)")
group.add_argument("--fail",    action="store_true", default=False,
                   help="Rejouer les épisodes échoués")
parser.add_argument("--ep",    type=int, default=None,
                    help="Index d'un épisode précis (0-based). Tous si absent.")
parser.add_argument("--delay", type=float, default=0.03,
                    help="Pause entre chaque step (s). Défaut : 0.03")
args = parser.parse_args()

# Défaut = success si rien n'est précisé
mode = "fail" if args.fail else "success"
data_dir = f"runs/reach_2dof/{mode}"

# ── Chargement des fichiers ────────────────────────────────────────────────────
pattern = os.path.join(data_dir, "ep_*.npz")
files   = sorted(glob.glob(pattern))

if not files:
    raise FileNotFoundError(
        f"Aucun fichier trouvé dans '{data_dir}'. "
        "Lancez d'abord runs_reach_2dof.py."
    )

if args.ep is not None:
    target_name = f"ep_{args.ep:03d}.npz"
    files = [f for f in files if os.path.basename(f) == target_name]
    if not files:
        raise FileNotFoundError(f"Épisode {args.ep:03d} introuvable dans {data_dir}.")

print(f"Replay [{mode.upper()}] — {len(files)} épisode(s) — reaching 2-DOF\n")

# ── Helpers géométrie ─────────────────────────────────────────────────────────
def fk(t1, t2, l1, l2):
    x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
    y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
    return np.array([x, y])

# ── Replay ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
plt.ion()

for file_idx, fpath in enumerate(files):
    ep_name = os.path.basename(fpath)
    d = np.load(fpath, allow_pickle=True)

    target  = d["target"]
    thetas  = d["thetas"]    # (T, 2)
    dists   = d["dists"]     # (T,)
    rewards = d["rewards"]
    l1      = float(d["l1"])
    l2      = float(d["l2"])
    epsilon = float(d["epsilon"])
    total_r = float(d["total_reward"])
    success = bool(d["success"])

    label = "✓ Succès" if success else "✗ Échec"
    print(f"[{file_idx+1}/{len(files)}]  {ep_name}  |  {label}  |  "
          f"steps={len(thetas)}  reward={total_r:+.1f}  "
          f"dist_finale={dists[-1]:.4f} m")

    for step_i, (t1, t2) in enumerate(thetas):
        ax.cla()

        j1  = np.array([l1 * np.cos(t1), l1 * np.sin(t1)])
        eff = fk(t1, t2, l1, l2)
        dist = dists[step_i]

        ax.plot([0, j1[0]], [0, j1[1]], 'r-', lw=4)
        ax.plot([j1[0], eff[0]], [j1[1], eff[1]], 'b-', lw=4)

        eff_color = "lime" if dist < epsilon else "red"
        ax.plot(eff[0], eff[1], "o", color=eff_color, markersize=10, label="End-effector")
        ax.plot(target[0], target[1], "go", markersize=12, label="Target")
        ax.add_patch(plt.Circle(target, epsilon,
                                color="green", fill=False, linestyle="--", lw=1.5))

        # Trajectoire passée de l'effecteur
        past_effs = np.array([fk(thetas[i, 0], thetas[i, 1], l1, l2)
                               for i in range(step_i + 1)])
        ax.plot(past_effs[:, 0], past_effs[:, 1], "b-", lw=1, alpha=0.3)

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_title(
            f"{ep_name}  [{mode}]  step {step_i+1}/{len(thetas)}\n"
            f"dist={dist:.3f} m  |  reward_cumulé={np.sum(rewards[:step_i+1]):+.1f}"
        )
        ax.legend(loc="upper left", fontsize=8)
        plt.pause(args.delay)

    time.sleep(0.5)

plt.ioff()
plt.show()
print("\nReplay terminé.")
