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

Vérifient la cinématique (FK, Jacobiens, pseudo-inverse) et le baseline analytique.

```bash
cd ~/Documents/transfer_learning/robot-robot
python3 -m direct_method.kinematics
python3 -m direct_method.action_mapper_baseline
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

## 6. Transfert push-ball (nécessite le PPO 2DoF avec son vec_normalize.pkl)

Sur ce PC, `data/models/ppo_pushball_2dof_2/` n'a **pas** de `vec_normalize.pkl` :
il faut d'abord ré-entraîner le PPO. Le script est configuré en **budget de
test : 15 M de timesteps** (au lieu de 150 M), 8 envs parallèles, LR en
décroissance linéaire ; il écrit dans `ppo_pushball_2dof_3` (sans écraser
l'ancien run) :

```bash
cd ~/Documents/transfer_learning/robot-robot
python3 -m agents.train.train_pushball_2dof
```

Puis comparer les trois mappers sur les mêmes seeds (adapter les chemins au run PPO produit) :

```bash
cd ~/Documents/transfer_learning/robot-robot
PPO_DIR=data/models/ppo_pushball_2dof_3   # <- adapter si besoin

python3 -m direct_method.transfer_pushball_3to2dof --mapper legacy      --episodes 100 --seed 0 \
    --model-path $PPO_DIR/best_model.zip --vecnorm-path $PPO_DIR/vec_normalize.pkl
python3 -m direct_method.transfer_pushball_3to2dof --mapper jacobian    --episodes 100 --seed 0 \
    --model-path $PPO_DIR/best_model.zip --vecnorm-path $PPO_DIR/vec_normalize.pkl
python3 -m direct_method.transfer_pushball_3to2dof --mapper conditioned --episodes 100 --seed 0 \
    --model-path $PPO_DIR/best_model.zip --vecnorm-path $PPO_DIR/vec_normalize.pkl
```

Avec rendu graphique : ajouter `--render` (et réduire `--episodes`).

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

### Bloc C — PPO push-ball 2DoF (budget test 15 M) + comparaison de transfert

```bash
cd ~/Documents/transfer_learning && \
source .venv/bin/activate && \
cd robot-robot && \
python3 -m agents.train.train_pushball_2dof && \
PPO_DIR=data/models/ppo_pushball_2dof_3 && \
python3 -m direct_method.transfer_pushball_3to2dof --mapper legacy      --episodes 100 --seed 0 \
    --model-path $PPO_DIR/best_model.zip --vecnorm-path $PPO_DIR/vec_normalize.pkl && \
python3 -m direct_method.transfer_pushball_3to2dof --mapper jacobian    --episodes 100 --seed 0 \
    --model-path $PPO_DIR/best_model.zip --vecnorm-path $PPO_DIR/vec_normalize.pkl && \
python3 -m direct_method.transfer_pushball_3to2dof --mapper conditioned --episodes 100 --seed 0 \
    --model-path $PPO_DIR/best_model.zip --vecnorm-path $PPO_DIR/vec_normalize.pkl && \
echo "=== TRANSFERT TERMINÉ ==="
```

> Note bloc C : le script PPO est en **budget de test (15 M timesteps)** et
> écrit dans `ppo_pushball_2dof_3`. Le `vec_normalize.pkl` doit être présent à
> côté de `best_model.zip` à la fin. Pour un run complet, remettre
> `TOTAL_TIMESTEPS = 150_000_000` dans `agents/train/train_pushball_2dof.py`
> (et incrémenter `run_id`).

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
| Modèles BC | `database-robot/data/models/bc_pushball_unified*/best_model.zip` |
