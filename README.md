# Transfert de politiques entre bras robotiques de morphologies différentes

Une politique de contrôle apprise par renforcement sur un bras **2 DoF** peut-elle piloter un bras
**3 DoF**, sans être réentraînée ? Le banc d'essai : deux bras planaires purement cinématiques, même
portée (3 m), mais des espaces d'états et d'actions incompatibles en dimension. Deux tâches —
*reaching* et *push-ball*.

L'approche ne modifie jamais la politique : elle insère un **state mapper** (état du robot cible →
état du robot source) et un **action mapper** (action de la politique → action du robot cible). La
politique croit contrôler son robot d'origine.

📄 **La documentation complète du projet est dans
[`PRESENTATION_PROJET.md`](PRESENTATION_PROJET.md)** — problème, idée directrice, diagnostic des
échecs, architecture retenue, résultats.

---

## Attribution

**Ce projet est le travail de [Jérôme Nortier](mailto:jerome.nortier@etu.uca.fr)** (rapport de M2,
`M2_ReportProjet_NORTIER.pdf`). Sur les 112 commits du dépôt, **108 sont de lui**.

**Ma contribution — Rigon Ajvazi — se limite à 4 commits, les 6 et 7 juillet 2026**, à titre de
regard extérieur sur un projet qui bloquait. Concrètement :

- une **revue technique des mappers** (`AVIS_TECHNIQUE_MAPPERS.md`,
  `CHANGEMENTS_ET_RAISONNEMENT.md`) : l'action mapper appris était aveugle à la posture, donc au
  Jacobien, et les données de correspondance étaient incohérentes en vitesse ;
- la **branche analytique** qui en découle — `kinematics.py` (cinématique directe, Jacobiens,
  pseudo-inverse amortie), `state_mapper_analytic.py` (cinématique inverse exacte 3 DoF → 2 DoF),
  `action_mapper_baseline.py` ;
- les **scripts d'évaluation et de visualisation** — `eval_mappers.py`,
  `eval_native_pushball.py`, `visualize_transfer.py`, `visualize_action_mapper.py` ;
- côté clonage de comportement : `train_bc.py` (capacité du réseau alignée sur l'expert, DAgger),
  `plot_bc_training.py`, `visualize_bc_pushball.py` ;
- le document de synthèse [`PRESENTATION_PROJET.md`](PRESENTATION_PROJET.md).

Cette contribution a été menée **avec l'assistance d'un agent d'IA** (voir `AGENTS.md`, commit
« ajout avis externe »). Le reste du dépôt — environnements, méthode directe, méthode latente
(VAE + PPO), politiques sources — **ne vient pas de moi**.

---

## Organisation

```
robot-robot/       transfert entre robots : envs, agents, direct_method, lsunn (méthode latente)
database-robot/    clonage de comportement à partir de démonstrations
PRESENTATION_PROJET.md   document de synthèse (le point d'entrée)
COMMANDES.md             comment lancer entraînements, transferts et évaluations
```

## Installation

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch tensorboard torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium numpy matplotlib pygame typing_extensions
pip install "stable-baselines3[extra]"
```

Les commandes d'entraînement, de transfert et d'évaluation sont dans
[`COMMANDES.md`](COMMANDES.md).
