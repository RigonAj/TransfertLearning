# Améliorations BC (database-robot) — 2026-07-07

Symptôme rapporté : le taux de réussite du behavior cloning push-ball 2DoF
reste inférieur à 40 %. Ce document donne le diagnostic, les modifications
apportées, et les résultats mesurés.

**Résultat final : 36,5 % → 91,2 % (500 épisodes indépendants), pour un
expert à 96,5 %.**

---

## 1. Diagnostic

Mesures initiales (200 épisodes, seed 0, max 150 pas, déterministe) :

| Politique | Réussite | Réseau |
|---|---|---|
| Expert `database/models/ppo_pushball_2dof_1` (source des démos) | 96,5 % | 256×256 |
| `bc_pushball_unified` (MSE) | 36,5 % | **64×64** |
| `bc_pushball_unified_nll` (NLL) | 52,5 % | **64×64** |

Les données ne sont **pas** en cause : `database/pushball_2dof/demonstrations.pkl`
contient 2 000 épisodes **tous réussis** (92 574 transitions, médiane 45 pas,
obs déjà normalisées VecNormalize de façon cohérente avec l'évaluation).

Deux causes réelles :

### 1.1 Sous-capacité : réseau étudiant 16× plus petit que l'expert

`train_bc.py` créait la politique par `PPO("MlpPolicy", env)` **sans
`policy_kwargs`** → réseau 64×64 (défaut SB3). L'expert cloné est un 256×256.
Le clone ne peut pas représenter la politique experte : il sous-apprend, tout
en affichant une loss de validation correcte (une MSE moyenne basse n'implique
pas une bonne politique aux états critiques, notamment au contact
effecteur-balle où la précision requise est de 0,2 m).

### 1.2 Dérive de distribution (compounding errors)

Défaut structurel du BC pur : l'étudiant n'est entraîné que sur les états
visités par l'expert. À la première déviation, il se retrouve sur des états
jamais vus, où son action est arbitraire → la déviation s'amplifie. La loss de
validation (mesurée sur la distribution de l'expert) est **aveugle** à ce
phénomène — c'est pourquoi « la loss était bonne mais le taux mauvais ».

Le remède standard est **DAgger** (Ross et al., 2011) : faire des rollouts de
l'ÉTUDIANT, demander à l'EXPERT l'action correcte sur ces états, agréger, et
ré-entraîner. Tout ce qu'il faut est disponible dans le dépôt : l'expert PPO et
ses stats VecNormalize.

---

## 2. Modifications

### 2.1 `train/train_bc.py`

1. **`--net-width 256` (nouveau défaut)** : `policy_kwargs` avec
   `net_arch pi/vf = [256, 256]`, même capacité que l'expert. C'est la
   correction au §1.1 — à elle seule : 36,5 % → 85 %.
2. **DAgger** (`--dagger-iters N`, défaut 0 = comportement BC pur inchangé) :
   après le BC initial, à chaque itération :
   - rollouts déterministes de l'étudiant (`--dagger-episodes`, défaut 25) ;
   - chaque état visité est étiqueté par l'action de l'expert
     (`--expert`, défaut `database/models/ppo_pushball_2dof_1/best_model.zip`) ;
   - agrégation au dataset et ré-entraînement (`--dagger-epochs`, défaut 20).
3. **Sélection du best model sur le taux de succès en rollout**
   (`--eval-episodes`, défaut 100, épisodes fixes pour comparer les
   itérations) et non plus sur la loss de validation — la loss ne mesure pas
   la dérive de distribution (§1.2). La phase BC initiale garde son early
   stopping sur la loss val.
4. `--max-steps` (défaut 150 = protocole d'enregistrement des démos) transmis
   à l'env de rollout/éval.

Le format de sortie est inchangé (SB3 `ppo.save` + `vec_normalize.pkl`),
compatible avec les scripts de test/replay existants.

### 2.2 Scripts de visualisation (nouveaux)

- **`test/visualize/visualize_bc_pushball.py`** : rollout en direct de
  n'importe quel modèle (`--model-dir`) — bras 2DoF, effecteur, balle avec sa
  trajectoire, cible avec rayon de succès, flèche de vitesse d'effecteur
  commandée, légende complète et verdict SUCCÈS/ÉCHEC. Avec
  `--compare-expert`, superpose la flèche de l'action que l'expert aurait
  prise sur les mêmes états, avec l'écart d'angle dans la légende : c'est le
  diagnostic visuel de la dérive de distribution (§1.2) — là où les flèches
  divergent, le clone quitte la distribution experte. `--save` écrit un GIF
  par épisode dans `data/plots/bc_rollouts/`.
- **`test/visualize/plot_bc_training.py`** : lit les logs TensorBoard d'un run
  et trace (1) les loss train/val de la phase BC et (2) le taux de succès par
  itération (BC pur puis D1…D10) avec la référence expert — la courbe sur
  laquelle le best model est choisi. Sortie :
  `<model-dir>/plots/training_summary.png`.

### 2.3 `test/test/bc_pushball.py`

- `--model-dir` (chemin direct, ex. `data/models/bc_pushball_dagger`),
  `--episodes`, `--max-steps` — l'ancien script n'acceptait que
  `--run_id` vers `data/models/bc_pushball_{id}/`, qui n'existe pas ;
- **bug corrigé** : `MAX_STEPS = 150` était déclaré mais jamais transmis à
  l'env, qui tournait au défaut (100 pas) ;
- ajout du bootstrap `sys.path` (comme `train_bc.py`) : le script était
  inlançable sans `PYTHONPATH=.`.

---

## 3. Résultats mesurés

Entraînement : `python3 train/train_bc.py --out data/models/bc_pushball_dagger
--dagger-iters 10 --eval-episodes 100` (MSE, 256×256, early stopping à
l'époque 246, ~10 min CPU).

| Étape | Succès (100 ép. d'éval fixes) |
|---|---|
| BC pur 64×64 (ancien `bc_pushball_unified`) | 36,5 %* |
| BC pur 256×256 | 85,0 % |
| DAgger itérations 1–5 | 75–85 % |
| DAgger itération 7 | 92,0 % |
| **DAgger itération 9 (best model sauvegardé)** | **95,0 %** |

\* mesuré sur 200 épisodes, protocole identique.

**Validation indépendante** (script officiel `test/test/bc_pushball.py`,
500 épisodes non vus pendant la sélection) :

```
bc_pushball_dagger : 91,2 % (456/500), 50 pas en moyenne sur succès
expert             : 96,5 %
```

Le clone est à ~5 points de l'expert. Note : les premières itérations DAgger
peuvent faire baisser temporairement le score (75 % à l'itération 1) — normal,
le mélange de distributions se stabilise ensuite.

Modèle final : `database-robot/data/models/bc_pushball_dagger/best_model.zip`
(+ `vec_normalize.pkl`). Les anciens modèles sont conservés.

---

## 4. Reproduire

```bash
cd ~/Documents/transfer_learning && source .venv/bin/activate && cd database-robot

# Entraînement complet BC + DAgger (~10 min CPU)
python3 train/train_bc.py --out data/models/bc_pushball_dagger --dagger-iters 10

# Validation indépendante (500 épisodes)
python3 test/test/bc_pushball.py --model-dir data/models/bc_pushball_dagger --episodes 500

# Référence : l'expert qui a généré les démos (attendu ~96 %)
python3 test/test/bc_pushball.py --model-dir database/models/ppo_pushball_2dof_1 --episodes 500

# Visualisation : rollouts avec légende (+ comparaison à l'expert), GIF avec --save
python3 test/visualize/visualize_bc_pushball.py --episodes 3 --compare-expert
MPLBACKEND=Agg python3 test/visualize/visualize_bc_pushball.py --episodes 5 --save --compare-expert

# Courbes d'entraînement (loss + succès par itération DAgger vs expert)
python3 test/visualize/plot_bc_training.py
```

---

## 5. Pistes pour aller plus loin

1. **Combler les ~5 points restants** : plus d'itérations DAgger
   (`--dagger-iters 20`), plus d'épisodes par itération, ou rollouts
   stochastiques de l'étudiant pour élargir la couverture d'états.
2. **Variante NLL + DAgger** (`--loss nll --dagger-iters 10`) : la NLL seule
   faisait déjà mieux que la MSE (52,5 % vs 36,5 % en 64×64) ; combinée au
   réseau 256 et à DAgger elle peut aider si la politique clonée sert ensuite
   d'initialisation PPO.
3. **BC 3DoF** : `database/pushball_3dof/demonstrations.pkl` existe ; le
   pipeline est prêt (`--env pushball_3dof`, `--expert` vers un PPO 3DoF).
4. À ±2–3 points près (variance binomiale à 100–500 épisodes), ne pas
   sur-interpréter les écarts entre itérations.
