# Commandes — entraînement, évaluation et visualisation

Toutes les commandes supposent le dépôt dans `~/Documents/transfer_learning`
et l'environnement virtuel `.venv` à la racine. Les scripts `direct_method`
se lancent depuis `robot-robot/`, les scripts BC depuis `database-robot/`.

> **Pressé ?** Va directement à la [section 8](#8-tout-en-une-seule-traite-copier-coller)
> pour copier-coller tous les blocs d'une traite.

---

## 0. Préparation (une fois par terminal)

```bash
cd ~/Documents/transfer_learning
source .venv/bin/activate
```

Si l'environnement n'existe pas encore (nouveau PC) :

```bash
cd ~/Documents/transfer_learning
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

---

## 1. Auto-tests (aucun entraînement requis, ~10 s)

Vérifient la cinématique (FK, Jacobiens, pseudo-inverse), le baseline
analytique d'action et le state mapper analytique (IK exacte 3→2).

```bash
cd ~/Documents/transfer_learning/robot-robot
python3 -m direct_method.kinematics
python3 -m direct_method.action_mapper_baseline
python3 -m direct_method.state_mapper_analytic
```

---

## 2. Entraînement des mappers (~5–10 min chacun, CPU ok, early stopping)

```bash
cd ~/Documents/transfer_learning/robot-robot

# State mapper avec losses cinématiques  ->  direct_method/runs/run_03_kin
python3 -m direct_method.train_states_mapper

# Action mapper conditionné (loss cartésienne)  ->  direct_method/runs/run_04_cond
python3 -m direct_method.train_actions_mapper
```

Smoke test rapide (5 époques, run jetable) :

```bash
DM_EPOCHS=5 DM_RUN_ID=essai python3 -m direct_method.train_states_mapper
DM_EPOCHS=5 DM_RUN_ID=essai python3 -m direct_method.train_actions_mapper
```

---

## 3. Évaluation offline (tableaux par zone + CSV + histogrammes)

```bash
cd ~/Documents/transfer_learning/robot-robot
python3 -m direct_method.eval_mappers
```

Sorties : console + `direct_method/runs/eval/metrics.csv` +
`direct_method/runs/eval/alignment_histograms.png`.

Variantes utiles :

```bash
# Évaluer seulement certains runs de state mapper
python3 -m direct_method.eval_mappers --run-states run_03_kin
python3 -m direct_method.eval_mappers --run-states run_01 run_03_kin   # comparaison ancien/nouveau
```

---

## 4. Visualisation — images PNG (non interactif, marche partout)

```bash
cd ~/Documents/transfer_learning/robot-robot

# Erreurs angle/norme vs rayon, pour chaque mapper (le "avant/après" du symptôme)
python3 -m direct_method.visualize_action_mapper --summary --mapper legacy      --direction 2to3
python3 -m direct_method.visualize_action_mapper --summary --mapper jacobian    --direction 2to3
python3 -m direct_method.visualize_action_mapper --summary --mapper conditioned --direction 2to3
python3 -m direct_method.visualize_action_mapper --summary --mapper jacobian    --direction 3to2
python3 -m direct_method.visualize_action_mapper --summary --mapper conditioned --direction 3to2

# Une frame précise (bras + flèches v_src / v_pred), sauvegardée en PNG
python3 -m direct_method.visualize_action_mapper --mapper jacobian --direction 2to3 --idx 1200 --save

# State mapper : reconstruction d'un état (PNG dans runs/run_03_kin/plots/)
MPLBACKEND=Agg python3 -m direct_method.visualize_velocity_mapper --idx 100 --save --run-id run_03_kin
```

Les PNG `--summary` et `--idx --save` de l'action mapper vont dans `direct_method/runs/eval/`.

## 5. Visualisation — fenêtres interactives (nécessite un écran)

```bash
cd ~/Documents/transfer_learning/robot-robot

# Animation action mapper (fermer la fenêtre pour arrêter)
python3 -m direct_method.visualize_action_mapper --mapper jacobian --direction 2to3 --delay 0.3

# Une frame action mapper à l'écran
python3 -m direct_method.visualize_action_mapper --mapper conditioned --idx 1200

# Animation state mapper
python3 -m direct_method.visualize_velocity_mapper --run-id run_03_kin --delay 0.3

# Une frame state mapper à l'écran
python3 -m direct_method.visualize_velocity_mapper --run-id run_03_kin --idx 100
```

---

## 6. Transfert push-ball (PPO 40 M entraîné le 2026-07-07)

Le PPO de référence est `data/models/ppo_pushball_2dof_4/` (40 M de steps).

> **ATTENTION : utiliser `ppo_pushball_final.zip`, PAS `best_model.zip`.**
> La sélection « best » de l'EvalCallback repose sur le reward moyen de
> 20 épisodes (bruité) et son `vec_normalize.pkl` est écrasé par la
> sauvegarde finale. Mesuré en natif : final = **99 %**, best_model
> (snapshot 22 M) = 59 %. Les défauts des scripts pointent déjà sur le final.

### 6.1 Plafond : évaluation native du PPO 2DoF (~1 min)

```bash
cd ~/Documents/transfer_learning/robot-robot
python3 -m direct_method.eval_native_pushball                  # attendu : 99 % (100 ép., seed 0)
python3 -m direct_method.eval_native_pushball \
    --model-path data/models/ppo_pushball_2dof_4/best_model.zip  # attendu : ~59 % (le piège)
```

### 6.2 Transfert : la meilleure config est le défaut

Défauts : state mapper `analytic` (IK exacte), action mapper `jacobian`,
PPO final 40 M, `--lam 0.02`.

```bash
cd ~/Documents/transfer_learning/robot-robot
python3 -m direct_method.transfer_pushball_3to2dof --seed 0                 # attendu : ~86 % (100 ép.)
python3 -m direct_method.transfer_pushball_3to2dof --episodes 300 --seed 0  # attendu : ~79 %
```

### 6.3 Comparaison complète des mappers (valeurs de référence, 100 ép., seed 0)

```bash
cd ~/Documents/transfer_learning/robot-robot
# state mapper analytique                                        attendu :
python3 -m direct_method.transfer_pushball_3to2dof --mapper jacobian    --seed 0   # 86 %
python3 -m direct_method.transfer_pushball_3to2dof --mapper conditioned --seed 0   # ~43 %
# state mapper appris run_03_kin (+ passthrough effecteur par défaut)
python3 -m direct_method.transfer_pushball_3to2dof --mapper conditioned --run-state run_03_kin --seed 0  # ~18 %
python3 -m direct_method.transfer_pushball_3to2dof --mapper jacobian    --run-state run_03_kin --seed 0  # ~4 %
python3 -m direct_method.transfer_pushball_3to2dof --mapper legacy      --run-state run_03_kin --seed 0  # ~3 %
# reproduire les anciens chiffres (sans passthrough) : ajouter --no-eff-passthrough
```

Balayage de l'amortissement (`--lam`) : 0,05 → 74 %, **0,02 → 79 %** (défaut),
0,01 → ~83 %, 0,005 → ~81 % (les valeurs à 100 ép. portent ±4 points).

À 100 épisodes l'incertitude est d'environ ±4 points : confirmer toute
conclusion sur 300+ épisodes ou plusieurs `--seed`.

### 6.4 (Optionnel) ré-entraîner le PPO

`agents/train/train_pushball_2dof.py` est configuré : 40 M timesteps,
sortie `ppo_pushball_2dof_4` — **incrémenter `run_id` avant de relancer**
pour ne pas écraser le run de référence :

```bash
cd ~/Documents/transfer_learning/robot-robot
python3 -m agents.train.train_pushball_2dof
```

## 6bis. Visualisation du transfert (bras réel + bras virtuel, avec légende)

Affiche le bras 3DoF réel, le bras 2DoF virtuel vu par la politique (pointillés),
la balle, la cible (rayon de succès) et les flèches de vitesse d'effecteur
commandée (PPO) vs exécutée (3DoF) — superposées si le mapping est bon.

```bash
cd ~/Documents/transfer_learning/robot-robot

# fenêtre interactive, 3 épisodes (config par défaut = la meilleure)
python3 -m direct_method.visualize_transfer --episodes 3

# comparer un mapper appris
python3 -m direct_method.visualize_transfer --episodes 3 --mapper conditioned --run-state run_03_kin

# GIF par épisode, sans écran -> direct_method/runs/eval/transfer_gifs/
MPLBACKEND=Agg python3 -m direct_method.visualize_transfer --episodes 5 --save
```

---

## 7. Behavior cloning (database-robot)

```bash
cd ~/Documents/transfer_learning/database-robot

# Version standard (MSE, best model choisi sur la validation)
python3 train/train_bc.py \
    --demos database/pushball_2dof/demonstrations.pkl \
    --vecnorm database/models/ppo_pushball_2dof_1/vec_normalize.pkl \
    --out data/models/bc_pushball_unified \
    --epochs 500

# Variante NLL (apprend aussi l'écart-type de la politique)
python3 train/train_bc.py --loss nll --out data/models/bc_pushball_unified_nll

# Smoke test rapide
python3 train/train_bc.py --epochs 5 --out /tmp/bc_smoke
```

Suivi TensorBoard :

```bash
tensorboard --logdir ~/Documents/transfer_learning/database-robot/data/models/bc_pushball_unified/tensorboard
```

---

## 8. Tout en une seule traite (copier-coller)

### Bloc A — pipeline mappers complet : auto-tests + entraînements + éval + tous les PNG

À coller tel quel dans le terminal (s'arrête à la première erreur grâce aux `&&`) :

```bash
cd ~/Documents/transfer_learning && \
source .venv/bin/activate && \
cd robot-robot && \
python3 -m direct_method.kinematics && \
python3 -m direct_method.action_mapper_baseline && \
python3 -m direct_method.train_states_mapper && \
python3 -m direct_method.train_actions_mapper && \
python3 -m direct_method.eval_mappers && \
python3 -m direct_method.visualize_action_mapper --summary --mapper legacy      --direction 2to3 && \
python3 -m direct_method.visualize_action_mapper --summary --mapper jacobian    --direction 2to3 && \
python3 -m direct_method.visualize_action_mapper --summary --mapper conditioned --direction 2to3 && \
python3 -m direct_method.visualize_action_mapper --summary --mapper jacobian    --direction 3to2 && \
python3 -m direct_method.visualize_action_mapper --summary --mapper conditioned --direction 3to2 && \
python3 -m direct_method.visualize_action_mapper --mapper jacobian --direction 2to3 --idx 1200 --save && \
MPLBACKEND=Agg python3 -m direct_method.visualize_velocity_mapper --idx 100 --save --run-id run_03_kin && \
echo "=== PIPELINE MAPPERS TERMINÉ — résultats dans direct_method/runs/ ==="
```

### Bloc B — behavior cloning (MSE puis NLL)

```bash
cd ~/Documents/transfer_learning && \
source .venv/bin/activate && \
cd database-robot && \
python3 train/train_bc.py \
    --demos database/pushball_2dof/demonstrations.pkl \
    --vecnorm database/models/ppo_pushball_2dof_1/vec_normalize.pkl \
    --out data/models/bc_pushball_unified --epochs 500 && \
python3 train/train_bc.py \
    --demos database/pushball_2dof/demonstrations.pkl \
    --vecnorm database/models/ppo_pushball_2dof_1/vec_normalize.pkl \
    --out data/models/bc_pushball_unified_nll --loss nll --epochs 500 && \
echo "=== BC TERMINÉ — modèles dans data/models/bc_pushball_unified*/ ==="
```

### Bloc C — vérification complète du transfert (PPO 40 M déjà entraîné, ~15 min)

Reproduit toutes les valeurs de référence du 2026-07-07
(`MODIFICATIONS_2026-07-07.md` §7) et génère les GIF de visualisation :

```bash
cd ~/Documents/transfer_learning && \
source .venv/bin/activate && \
cd robot-robot && \
python3 -m direct_method.state_mapper_analytic && \
python3 -m direct_method.eval_native_pushball && \
python3 -m direct_method.eval_native_pushball --model-path data/models/ppo_pushball_2dof_4/best_model.zip && \
python3 -m direct_method.transfer_pushball_3to2dof --mapper jacobian    --seed 0 && \
python3 -m direct_method.transfer_pushball_3to2dof --mapper conditioned --seed 0 && \
python3 -m direct_method.transfer_pushball_3to2dof --mapper conditioned --run-state run_03_kin --seed 0 && \
python3 -m direct_method.transfer_pushball_3to2dof --mapper jacobian    --run-state run_03_kin --seed 0 && \
python3 -m direct_method.transfer_pushball_3to2dof --mapper legacy      --run-state run_03_kin --seed 0 && \
python3 -m direct_method.transfer_pushball_3to2dof --episodes 300 --seed 0 && \
MPLBACKEND=Agg python3 -m direct_method.visualize_transfer --episodes 5 --save && \
echo "=== VÉRIFICATION TERMINÉE — attendu : natif 99 % / 59 %, transfert 86, ~43, ~18, ~4, ~3 %, puis ~79 % sur 300 ép. ==="
```

> Valeurs attendues (seed 0 ; ±4 points à 100 épisodes) : natif final 99 %,
> natif best_model 59 % (le piège), puis analytic+jacobian 86 %,
> analytic+conditioned ~43 %, run_03_kin+conditioned ~18 %,
> run_03_kin+jacobian ~4 %, legacy ~3 %, et ~79 % sur 300 épisodes.
> Les GIF sont dans `direct_method/runs/eval/transfer_gifs/`.

### Bloc D — ré-entraînement PPO 2DoF (optionnel, 40 M timesteps)

**Incrémenter `run_id` dans `agents/train/train_pushball_2dof.py` d'abord**
(le run de référence `ppo_pushball_2dof_4` ne doit pas être écrasé), puis :

```bash
cd ~/Documents/transfer_learning && \
source .venv/bin/activate && \
cd robot-robot && \
python3 -m agents.train.train_pushball_2dof && \
echo "=== PPO TERMINÉ — évaluer ppo_pushball_final.zip (pas best_model) avec eval_native_pushball ==="
```

---

## Où trouver les résultats

| Sortie | Emplacement |
|---|---|
| Modèles state mapper | `robot-robot/direct_method/runs/run_03_kin/models/` |
| Modèles action mapper | `robot-robot/direct_method/runs/run_04_cond/models/` |
| Courbes d'entraînement | `robot-robot/direct_method/runs/run_0*/plots/loss_curves.png` |
| Logs CSV (par composante de loss) | `robot-robot/direct_method/runs/run_0*/logs/training_log.csv` |
| Métriques d'évaluation | `robot-robot/direct_method/runs/eval/metrics.csv` |
| Graphiques avant/après symptôme | `robot-robot/direct_method/runs/eval/action_mapper_*_summary.png` |
| GIF de visualisation du transfert | `robot-robot/direct_method/runs/eval/transfer_gifs/` |
| PPO 2DoF de référence (40 M) | `robot-robot/data/models/ppo_pushball_2dof_4/ppo_pushball_final.zip` |
| Modèles BC | `database-robot/data/models/bc_pushball_unified*/best_model.zip` |
