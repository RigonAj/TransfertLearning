"""Courbes d'entraînement BC : loss train/val + taux de succès par itération DAgger.

Lit les logs TensorBoard écrits par train/train_bc.py et produit un PNG à deux
panneaux :
  - gauche : loss d'entraînement et de validation par époque (phase BC) ;
  - droite : taux de succès en rollout — point 0 = BC pur, points suivants =
    itérations DAgger — avec la référence de l'expert en pointillés.

Usage :
    cd database-robot
    python3 test/visualize/plot_bc_training.py                       # modèle DAgger par défaut
    python3 test/visualize/plot_bc_training.py --model-dir data/models/bc_pushball_unified_v2
Sortie : <model-dir>/plots/training_summary.png (+ fenêtre si écran).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]


def load_scalars(tb_dir):
    acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        out[tag] = (np.array([e.step for e in events]),
                    np.array([e.value for e in events]))
    return out


def main():
    p = argparse.ArgumentParser(description="Courbes d'entraînement BC/DAgger.")
    p.add_argument("--model-dir", default="data/models/bc_pushball_dagger")
    p.add_argument("--expert-rate", type=float, default=96.5,
                   help="taux de succès de l'expert en %% (référence ; mesuré "
                        "sur ppo_pushball_2dof_1 : 96,5)")
    p.add_argument("--save-only", action="store_true", help="ne pas ouvrir de fenêtre")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    tb_dir = model_dir / "tensorboard"
    if not tb_dir.exists():
        sys.exit(f"Pas de logs TensorBoard dans {tb_dir} — entraîner d'abord avec train_bc.py")

    scalars = load_scalars(tb_dir)

    fig, (ax_loss, ax_succ) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panneau loss ──────────────────────────────────────────────────
    for tag, style, label in [("loss/epoch_train", "-", "loss entraînement"),
                              ("loss/epoch_val", "-", "loss validation")]:
        if tag in scalars:
            steps, vals = scalars[tag]
            ax_loss.plot(steps, vals, style, lw=1.8, label=label)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Époque (phase BC)")
    ax_loss.set_ylabel("Loss (échelle log)")
    ax_loss.set_title("Phase BC : loss par époque\n(best model BC = min de la loss val)")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    # ── Panneau succès ────────────────────────────────────────────────
    if "success_rate" in scalars:
        steps, vals = scalars["success_rate"]
        vals = vals * 100.0
        ax_succ.plot(steps, vals, "-o", color="#1f77b4", lw=2,
                     label="politique clonée")
        for s, v in zip(steps, vals):
            ax_succ.annotate(f"{v:.0f}%", (s, v), textcoords="offset points",
                             xytext=(0, 7), ha="center", fontsize=8)
        best = vals.max()
        ax_succ.axhline(args.expert_rate, color="#9B5DE5", ls="--", lw=1.8,
                        label=f"expert ({args.expert_rate:.1f} %)")
        ax_succ.set_xticks(steps)
        ax_succ.set_xticklabels(["BC pur"] + [f"D{int(s)}" for s in steps[1:]])
        ax_succ.set_ylim(0, 100)
        ax_succ.set_xlabel("Itération (BC pur puis DAgger)")
        ax_succ.set_ylabel("Taux de succès en rollout (%)")
        ax_succ.set_title(f"Succès par itération — meilleur : {best:.0f} %\n"
                          "(le best_model final est choisi sur cette courbe)")
        ax_succ.grid(True, alpha=0.3)
        ax_succ.legend(loc="lower right")
    else:
        ax_succ.text(0.5, 0.5, "Pas de scalaire success_rate\n"
                     "(run antérieur au 2026-07-07 ?)",
                     ha="center", va="center", transform=ax_succ.transAxes)

    fig.suptitle(f"Behavior cloning — {model_dir.name}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = model_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure → {out_path}")

    if not args.save_only and matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
