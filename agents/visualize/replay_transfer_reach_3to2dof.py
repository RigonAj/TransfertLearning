"""
replay_transfer_reach_3to2dof.py
Rejoue les épisodes sauvegardés par runs_transfer_reach_3to2dof.py.
Politique 3-DOF transférée, exécutée dans l'environnement 2-DOF.

Usage :
  python replay_transfer_reach_3to2dof.py              # succès (défaut)
  python replay_transfer_reach_3to2dof.py --fail
  python replay_transfer_reach_3to2dof.py --fail --ep 3
  python replay_transfer_reach_3to2dof.py --success --delay 0.05
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
parser = argparse.ArgumentParser(description="Replay transfer reaching 3→2 DOF")
group = parser.add_mutually_exclusive_group()
group.add_argument("--success", action="store_true", default=False)
group.add_argument("--fail",    action="store_true", default=False)
parser.add_argument("--ep",    type=int,   default=None)
parser.add_argument("--delay", type=float, default=0.03)
args = parser.parse_args()

mode     = "fail" if args.fail else "success"
data_dir = f"runs/transfer_reach_3to2dof/{mode}"

# ── Chargement ────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(data_dir, "ep_*.npz")))
if not files:
    raise FileNotFoundError(
        f"Aucun fichier dans '{data_dir}'. Lancez runs_transfer_reach_3to2dof.py d'abord."
    )
if args.ep is not None:
    target_name = f"ep_{args.ep:03d}.npz"
    files = [f for f in files if os.path.basename(f) == target_name]
    if not files:
        raise FileNotFoundError(f"Épisode {args.ep:03d} introuvable.")

print(f"Replay [{mode.upper()}] — {len(files)} épisode(s) — transfer reaching 3→2 DOF\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fk2(t1, t2, l1, l2):
    x = l1*np.cos(t1) + l2*np.cos(t1+t2)
    y = l1*np.sin(t1) + l2*np.sin(t1+t2)
    return np.array([x, y])

# ── Replay ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
plt.ion()

for file_idx, fpath in enumerate(files):
    ep_name = os.path.basename(fpath)
    d = np.load(fpath, allow_pickle=True)

    target  = d["target"]
    thetas  = d["thetas"]    # (T, 2)
    dists   = d["dists"]
    rewards = d["rewards"]
    l1      = float(d["l1"])
    l2      = float(d["l2"])
    epsilon = float(d["epsilon"])
    total_r = float(d["total_reward"])
    success = bool(d["success"])

    label = "✓ Succès" if success else "✗ Échec"
    print(f"[{file_idx+1}/{len(files)}]  {ep_name}  |  {label}  |  "
          f"steps={len(thetas)}  reward={total_r:+.1f}  dist_finale={dists[-1]:.4f} m")

    for step_i, (t1, t2) in enumerate(thetas):
        ax.cla()

        j1  = np.array([l1*np.cos(t1), l1*np.sin(t1)])
        eff = fk2(t1, t2, l1, l2)
        dist = dists[step_i]

        ax.plot([0, j1[0]], [0, j1[1]], 'r-', lw=4)
        ax.plot([j1[0], eff[0]], [j1[1], eff[1]], 'b-', lw=4)

        eff_color = "lime" if dist < epsilon else "red"
        ax.plot(eff[0], eff[1], "o", color=eff_color, markersize=10, label="End-effector")
        ax.plot(target[0], target[1], "go", markersize=12, label="Target")
        ax.add_patch(plt.Circle(target, epsilon,
                                color="green", fill=False, linestyle="--", lw=1.5))

        past_effs = np.array([fk2(thetas[i,0], thetas[i,1], l1, l2)
                               for i in range(step_i + 1)])
        ax.plot(past_effs[:, 0], past_effs[:, 1], "b-", lw=1, alpha=0.3)

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_title(
            f"[Transfer 3→2 DOF]  {ep_name}  [{mode}]  step {step_i+1}/{len(thetas)}\n"
            f"dist={dist:.3f} m  |  reward_cumulé={np.sum(rewards[:step_i+1]):+.1f}"
        )
        ax.legend(loc="upper left", fontsize=8)
        plt.pause(args.delay)

    time.sleep(0.5)

plt.ioff()
plt.show()
print("\nReplay terminé.")
