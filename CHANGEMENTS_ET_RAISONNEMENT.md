# Changements et raisonnement — mappers robot-robot et BC database-robot

Date : 2026-07-06. Ce document explique en détail chaque changement apporté au
dépôt, le raisonnement qui y a conduit, et comment reproduire/interpréter les
résultats.

---

## 1. Le diagnostic : pourquoi l'action mapper échouait

### 1.1 Le symptôme observé

> Lorsque l'effecteur est loin de (0, 0) avec le bras presque tendu, les
> vitesses prédites sont différentes mais alignées ; lorsque l'effecteur se
> rapproche de (0, 0) avec le bras plié, c'est la différence d'angle entre les
> vitesses prédites qui augmente mais la norme semble être la même.

### 1.2 La cause structurelle

La vitesse cartésienne de l'effecteur d'un bras est :

```
v_eff = J(θ) @ dθ
```

où le Jacobien `J(θ)` dépend des angles articulaires. **La même commande `dθ`
produit des vitesses d'effecteur complètement différentes selon la posture.**

Or l'ancien `action_mapper` (`direct_method/train_actions_mapper.py`) apprenait
uniquement :

```
[dθ1, dθ2]  →  [dθ1, dθ2, dθ3]
```

sans jamais voir les angles. Il ne pouvait donc pas savoir quel Jacobien
s'applique, ni côté source ni côté cible. Il apprenait au mieux la relation
*moyenne* sur le dataset — d'où :

- **bras tendu (r → 3 m)** : près d'une singularité, le gain du Jacobien varie
  énormément → la direction moyenne reste à peu près bonne mais la **norme**
  est fausse ;
- **bras plié près du centre (r → 0)** : les colonnes du Jacobien tournent très
  vite avec θ, plusieurs postures donnent des positions proches mais des
  directions locales opposées → la norme moyenne est plausible mais la
  **direction** est fausse.

C'est exactement le symptôme rapporté. Le graphique
`direct_method/runs/eval/action_mapper_legacy_2to3_summary.png` le montre
quantitativement : erreur d'angle médiane ~28° près du centre décroissant vers
~5° au bord, erreur de norme ~0,15–0,2 m/s partout.

### 1.3 Deux découvertes supplémentaires dans les données

En instrumentant les fichiers `direct_method/trajectories/robot_{2,3}dofs.txt`
(97 435 lignes appariées), deux faits importants sont apparus :

**a) Les colonnes « effecteur » ne sont pas en mètres.** Les 2 dernières
colonnes sont identiques dans les deux fichiers et encodées en coordonnées
fenêtre `(eff + 5) / 10` (vérifié numériquement : `eff = 10·col − 5` reproduit
`FK(θ)` à 7·10⁻⁶ près). L'ancien pipeline les normalisait par `max_reach = 3`,
produisant des features dans [0,07 ; 0,27]… alors qu'au moment du transfert les
environnements fournissent `eff / 3` dans [−1 ; 1]. Le state mapper legacy
était donc entraîné sur une distribution d'entrée différente de celle qu'il
voit en production.

**b) Les positions sont parfaitement alignées, pas les vitesses.** Ligne à
ligne, `‖FK2(θ_2dof) − FK3(θ_3dof)‖ = 0` exactement (la correspondance de
posture a été générée analytiquement), mais `‖v2 − v3‖` a une médiane de
**0,26 m/s** (p90 = 0,77). Autrement dit, les cibles articulaires 3DoF du
dataset ne reproduisent PAS la vitesse d'effecteur du 2DoF. Ce n'est pas un
détail : l'erreur résiduelle du mapper legacy (0,2632 m/s en médiane) est
exactement l'écart de vitesse du dataset — le réseau apprenait fidèlement des
cibles cinématiquement incohérentes.

**Conséquence :** une MSE articulaire vers ces cibles ne peut pas donner un bon
mapper, même avec un réseau parfait. Il faut viser directement la grandeur
physique commune : la vitesse cartésienne de l'effecteur **source**.

---

## 2. L'architecture de la solution

Trois niveaux, du plus sûr au plus flexible :

1. **Cinématique exacte partagée** (`kinematics.py`) : FK, Jacobiens,
   pseudo-inverse amortie — la vérité physique, testée contre les envs.
2. **Baseline analytique sans apprentissage** (`action_mapper_baseline.py`) :
   `dθ_tgt = J_tgt⁺(θ_tgt) @ (J_src(θ_src) @ dθ_src)`. Il fixe le niveau de
   performance de référence et prouve que le problème est cinématique.
3. **Réseau conditionné avec loss cartésienne** (`train_actions_mapper.py`) :
   garde l'approche apprise du projet, mais avec les bonnes entrées (θ_src,
   θ_tgt) et le bon objectif (v_eff).

Résultats offline mesurés (validation 9 800 échantillons, direction 2DoF→3DoF,
médianes ; direction 3→2 similaire) :

| Mapper | ‖v_pred − v_src‖ | Erreur de norme | Erreur d'angle | MSE artic. |
|---|---|---|---|---|
| legacy (run_02) | 0,263 m/s | 0,174 m/s | 8,94° | 0,0003 |
| **jacobian (baseline)** | **0,0010 m/s** | 0,0009 m/s | **0,024°** | 0,0067 |
| conditioned (run_04_cond) | 0,0045 m/s | 0,0017 m/s | 0,30° | 0,0035 |

Lecture :

- les deux nouvelles approches réduisent l'erreur de vitesse d'un facteur
  **60 à 260** par rapport au legacy, dans les deux régimes du symptôme
  (angle près du centre : 19,3° → 0,03°/0,47° ; norme au bord :
  0,15 m/s → 0,001/0,004 m/s) ;
- le baseline Jacobien est meilleur en médiane, mais le mapper conditionné est
  **plus robuste dans les pires cas près des singularités** : en zone centre,
  p90 de ‖v_pred − v_src‖ = 0,040 m/s contre 0,349 m/s pour la pinv amortie
  (l'amortissement λ dégrade la solution quand J_tgt devient singulier, alors
  que le réseau a appris à contourner ces configurations) ;
- noter la MSE articulaire : le legacy est excellent sur ce critère (0,0003)
  tout en étant le pire en espace effecteur — preuve définitive que la MSE
  articulaire vers ce dataset n'est pas le bon objectif (cf. §1.3.b).

---

## 3. Changements fichier par fichier (`robot-robot/direct_method/`)

### 3.1 `kinematics.py` (nouveau)

Source unique de vérité pour la cinématique et les constantes (`L_2DOF`,
`L_3DOF`, `OMEGA_MAX`, `MAX_REACH`, `DT`). Contient :

- `fk`, `jacobian`, `eff_vel` génériques (torch, batchés, **différentiables** —
  utilisables dans les losses), et les raccourcis `fk_2dof`, `jacobian_3dof`… ;
- les mêmes en NumPy (`*_np`) pour les scripts d'évaluation/transfert ;
- `damped_pinv(J, lam)` : pseudo-inverse amortie
  `Jᵀ(JJᵀ + λ²I)⁻¹`. L'amortissement borne le gain près des singularités
  (bras tendu) — une pinv brute y exploserait ;
- helpers de normalisation (θ/π, dθ/ω_max, eff/max_reach) et `split_state`.

Auto-test intégré (`python3 -m direct_method.kinematics`) : FK et v_eff
comparés aux méthodes des envs (erreur < 10⁻⁷), Jacobiens contre différences
finies (< 10⁻⁹), reconstruction `J @ (J⁺ @ v) ≈ v` hors singularités.

*Pourquoi :* les FK/Jacobiens étaient dupliqués (envs, visualiseurs) et aucune
version différentiable n'existait pour les losses. Toute divergence entre
copies aurait faussé l'entraînement silencieusement.

### 3.2 `action_mapper_baseline.py` (nouveau)

`JacobianActionMapper(direction, lam, null_space_gain)` :

```
v      = J_src(θ_src) @ dθ_src
dθ_tgt = damped_pinv(J_tgt(θ_tgt)) @ v          [solution de norme minimale]
       + (I − J⁺J) @ z                          [optionnel, 3DoF uniquement]
```

Le terme de null-space `z = −gain·θ_tgt` permet de ramener la posture 3DoF vers
une configuration neutre **sans modifier la vitesse de l'effecteur** (le
projecteur `I − J⁺J` garantit que z n'agit que dans le noyau du Jacobien).
Par défaut `gain = 0` : solution de norme minimale.

*Pourquoi :* aucun entraînement requis, interprétable, et il résout la
redondance 2→3 proprement. Si un réseau ne bat pas ce baseline, le réseau
n'apporte rien. Auto-test intégré (reproduction de v_eff, invariance du
null-space).

### 3.3 `mapper_models.py` (étendu)

Ajout de `ActionMapperConditionedMLP(nq_src, nq_tgt, hidden=256)` :

- entrées : `dθ_src` normalisé + `θ_src` + `θ_tgt` (normalisés /π) ;
- featurisation **sin/cos dans `forward`** : représentation continue à la
  discontinuité ±π (un angle normalisé brut saute de −1 à +1 au passage de
  ±180°, ce qui force le réseau à apprendre une discontinuité), et c'est
  exactement la forme sous laquelle θ entre dans les Jacobiens ;
- sortie : `dθ_tgt` normalisé.

Les anciennes classes `StateMapperMLP` et `ActionMapperMLP` sont conservées
pour recharger les runs existants (run_01, run_02).

*Alternative rejetée :* donner directement le Jacobien aplati en entrée. Ça
marche aussi, mais sin/cos suffit (le Jacobien est une fonction lisse de
sin/cos des angles cumulés) et garde le modèle petit.

### 3.4 `dataset.py` (corrigé et étendu)

- `TrajectoriesTrainingDataset` : les colonnes effecteur sont désormais
  **recalculées par FK(θ)** puis normalisées /3 — correction du bug d'encodage
  fenêtre décrit en §1.3.a. Le format de sortie (6D/8D) est inchangé.
- `TrajectoriesTrainingActionDataset` : retourne `(vel_r1, q_r1, vel_r2, q_r2)`
  au lieu des seules vitesses (reprise de l'idée de `direct_method_2`,
  nécessaire au conditionnement).
- `block_split(dataset, fracs, block=200, seed)` : split train/val/test par
  **blocs contigus** de 200 lignes. `random_split` par lignes mettait des
  timesteps quasi identiques de part et d'autre du split → validation
  artificiellement optimiste (fuite). C'est en partie pourquoi le mapper legacy
  « semblait bon » à l'entraînement.

### 3.5 `train_actions_mapper.py` (réécrit)

Entraîne `ActionMapperConditionedMLP` dans les deux directions avec la loss :

```
loss = 1.0  · Huber(v_pred − v_src)              # objectif principal
     + 0.5  · (1 − cos(v_pred, v_src))           # direction
     + 0.25 · | ‖v_pred‖ − ‖v_src‖ |             # norme
     + 0.1  · MSE(dθ_pred, dθ_dataset)           # régularisateur (faible !)
     + 1e-4 · ‖dθ_pred‖²                          # préférence norme minimale
```

avec `v_src = J_src(θ_src)@dθ_src` et `v_pred = J_tgt(θ_tgt)@dθ_pred`, calculés
en unités réelles par la cinématique différentiable.

Choix importants :

- **la MSE articulaire a un poids faible (0,1)** : cf. §1.3.b, les cibles du
  dataset sont incohérentes en vitesse (écart médian 0,26 m/s). Un poids fort
  tirerait le réseau vers cette erreur. Elle reste utile comme prior de
  redondance (rester près des postures démontrées) ;
- la loss d'angle n'est appliquée que si `‖v_src‖ > 0,05 m/s` (la direction
  d'une vitesse quasi nulle n'a pas de sens et injecterait du bruit) ;
- LR 10⁻³ (l'ancien 3·10⁻³ était élevé pour Adam), `ReduceLROnPlateau`,
  early stopping (patience 15) sur la **loss cartésienne de validation**,
  seed fixé, sélection du meilleur modèle sur validation ;
- sorties : `runs/run_04_cond/` (modèles `action_mapper_2to3.pt`/`_3to2.pt`,
  `config.json` avec tous les hyperparamètres, CSV par composante de loss,
  courbes). `DM_EPOCHS=5 python3 -m …` pour un smoke test rapide.

### 3.6 `train_states_mapper.py` (réécrit)

- Importe le dataset et `StateMapperMLP` partagés (l'ancienne version
  redéfinissait localement les deux — divergences garanties à terme).
- Loss enrichie :

```
loss = MSE(état_pred, état_cible)
     + 0.5  · ‖FK(θ_pred) − eff_src‖²            # même position d'effecteur
     + 0.25 · ‖J(θ_pred)@dθ_pred − v_src‖²       # même vitesse d'effecteur
     + 0.1  · ‖map_back(map(s)) − s‖²            # cohérence cycle
```

*Pourquoi la loss FK :* une MSE sur les angles pondère chaque articulation
également, alors qu'une petite erreur sur θ1 déplace l'effecteur beaucoup plus
qu'une erreur sur θ3. La loss FK pénalise l'effet cartésien réel.
*Pourquoi la loss cycle :* les deux directions sont entraînées conjointement
(un seul optimiseur) ; le cycle stabilise la bijection approximative entre les
deux espaces d'états.

- Mêmes améliorations d'infrastructure que 3.5 (split par blocs, scheduler,
  early stopping, seed, CSV par composante). Sorties : `runs/run_03_kin/`,
  noms de fichiers modèles inchangés (compatibilité transfert/visualiseurs).

### 3.7 `eval_mappers.py` (nouveau)

Évaluation offline systématique sur le split de validation :

- action mappers `legacy` / `jacobian` / `conditioned`, deux directions ;
- state mappers (run_01 et run_03_kin) ;
- métriques : ‖v_pred − v_src‖, erreur de norme, erreur d'angle,
  MSE articulaire, ‖FK(θ_pred) − eff_src‖ ;
- **ventilation par zone de rayon** (centre < 1 m / milieu / bord > 2,5 m) —
  la ventilation du symptôme ;
- diagnostic d'alignement des trajectoires (position et vitesse, cf. §1.3.b)
  avec histogrammes ;
- sorties : tableaux console (médiane/p90), `runs/eval/metrics.csv`, PNG.

*Pourquoi :* sans métriques en espace effecteur, on optimise une loss qui ne
mesure pas le problème. C'est l'outil de comparaison honnête entre approches.

### 3.8 `transfer_pushball_3to2dof.py` (corrigé)

Trois bugs corrigés :

1. **Dimensions** : `R1_STATE_DIM = 4, R2_STATE_DIM = 6` → **6 et 8**. L'ancien
   script construisait un `StateMapperMLP(6→4)` et tentait d'y charger des
   poids 8→6 (crash ou comportement indéfini selon la version de torch).
2. **Normalisation des observations** : la politique PPO 2DoF a été entraînée
   sur des obs passées par `VecNormalize`. L'ancien script chargeait le
   VecNormalize 2DoF (shape 10) sur l'env 3DoF (shape 12) puis le désactivait
   (`norm_obs = False`) et donnait des obs brutes à la politique. Corrigé : le
   VecNormalize 2DoF est chargé sur un env 2DoF factice et sert uniquement à
   normaliser l'obs mappée juste avant `model.predict`.
3. **Hack d'échelle** : `action_mapper(action.clamp(-2,2) * 0.5)` — un facteur
   arbitraire qui compensait empiriquement le mapper défaillant. Supprimé : les
   mappers corrects n'en ont pas besoin.

Ajouts : `--mapper {legacy, jacobian, conditioned}`, `--seed`, `--episodes`,
`--run-state`, `--render`. Permet la comparaison A/B sur les mêmes graines.

Flux corrigé complet :

```
obs 3DoF brute (12D) = [bras 8D | balle 2D | cible 2D]
  → state_mapper_r2_to_r1(bras 8D) → bras 2DoF 6D
  → concat + VecNormalize 2DoF → PPO 2DoF → action_2
  → action mapper (θ2 mappé, θ3 réel) → action_3 → env 3DoF
```

### 3.9 Visualiseurs

- `visualize_velocity_mapper.py` : utilise désormais `kinematics.py` (au lieu
  de copies locales de FK/Jacobiens) et instancie `StateMapperMLP` — l'ancien
  code chargeait les poids du state mapper dans un `ActionMapperMLP` (même
  architecture par coïncidence, mais nom trompeur). La normalisation d'entrée
  recalcule l'effecteur par FK (bug §1.3.a). Défaut : `--run-id run_03_kin`.
- `visualize_action_mapper.py` (nouveau) : reproduit le protocole visuel du
  symptôme (bras côte à côte, flèches v_src/v_pred, erreur d'angle et normes
  affichées) pour n'importe quel mapper, plus un mode `--summary` qui trace
  erreur d'angle et de norme **en fonction du rayon** sur la validation.
  Les PNG `runs/eval/action_mapper_{legacy,jacobian}_2to3_summary.png` générés
  montrent le avant/après.

### 3.10 `direct_method_2/` — statut

Non modifié (archive). Sa bonne idée — conditionner par la configuration — est
intégrée et généralisée dans `direct_method` (conditionnement par θ_src **et**
θ_tgt + loss cartésienne, ce que `direct_method_2` n'avait pas : il gardait une
MSE articulaire pure, écrivait ses sorties dans `direct_method/runs/` et son
visualiseur était incompatible avec la signature de son propre modèle).

---

## 4. Partie BC — `database-robot/train/train_bc.py` (nouveau)

### 4.1 Verdict sur les deux anciennes versions

Les deux scripts font fondamentalement la même chose : MSE entre la moyenne de
la distribution de la politique SB3 et l'action démontrée.

| | `train_behavior_cloning.py` | `train_bc_imitation.py` |
|---|---|---|
| Boucle données | DataLoader propre | batching manuel **bugué** : `n_batches = size//bs + 1` → si `size % bs == 0`, le dernier batch est vide (crash potentiel) et la moyenne de loss est biaisée |
| Lib `imitation` | non utilisée | importée, `Transitions` construites… puis jamais utilisées (la boucle est manuelle) |
| Sélection du modèle | loss **train** (biais overfitting) | loss **train** (idem) |
| Validation | aucune | aucune |

**`train_behavior_cloning.py` est la meilleure base.** Les deux restent en
place ; `train_bc.py` devient la référence.

### 4.2 Ce que le script unifié ajoute

- **Split train/val par épisode** (défaut 90/10) : deux transitions d'un même
  épisode sont fortement corrélées ; les répartir aléatoirement entre train et
  val rendrait la validation optimiste. Best model choisi sur la **loss val**.
- **Deux losses** : `--loss mse` (défaut, comportement historique) et
  `--loss nll` (log-vraisemblance gaussienne + terme d'entropie, comme le BC de
  la lib `imitation`) qui apprend aussi `log_std` — utile si la politique
  clonée est utilisée en mode stochastique ou comme initialisation PPO.
- **Contrôle de normalisation** : à l'ouverture des démos, le script vérifie si
  les obs sont déjà normalisées VecNormalize (std ≈ 1, plage > ±1,5). Pour
  `database/pushball_2dof/demonstrations.pkl` : elles le sont (std = 1,004,
  max = 3,07) → l'entraînement historique était cohérent, ne PAS utiliser
  `--normalize-obs` avec ces démos. L'option existe pour des démos brutes.
- Scheduler `ReduceLROnPlateau`, early stopping (`--patience 25`), gradient
  clipping, seed, checkpoints périodiques, TensorBoard (train/val/LR).
- CLI complète (`--env pushball_3dof` pour le 3DoF, chemins configurables).
- Sauvegarde inchangée (`ppo.save` + `vec_normalize.pkl`) : compatible
  `test/visualize/replay_bc_pushball.py`.

---

## 5. Comment reproduire (dans l'ordre)

Depuis `robot-robot/`, venv activé :

```bash
# 1. Auto-tests (aucun entraînement requis)
python3 -m direct_method.kinematics
python3 -m direct_method.action_mapper_baseline

# 2. Entraînements (CPU ok, ~2 s/époque, early stopping)
python3 -m direct_method.train_states_mapper      # -> runs/run_03_kin
python3 -m direct_method.train_actions_mapper     # -> runs/run_04_cond
#    (smoke test rapide : DM_EPOCHS=5 DM_RUN_ID=essai python3 -m …)

# 3. Évaluation offline comparative
python3 -m direct_method.eval_mappers             # tableaux + runs/eval/metrics.csv

# 4. Visualisation
python3 -m direct_method.visualize_action_mapper --summary --mapper jacobian
python3 -m direct_method.visualize_action_mapper --summary --mapper conditioned
python3 -m direct_method.visualize_action_mapper --mapper jacobian --idx 1200
python3 -m direct_method.visualize_velocity_mapper --idx 100

# 5. Transfert push-ball (nécessite data/models/ppo_pushball_2dof_1/ entraîné)
python3 -m direct_method.transfer_pushball_3to2dof --mapper jacobian --seed 0
python3 -m direct_method.transfer_pushball_3to2dof --mapper conditioned --seed 0
python3 -m direct_method.transfer_pushball_3to2dof --mapper legacy --seed 0
```

Depuis `database-robot/` :

```bash
python3 train/train_bc.py --epochs 500            # MSE, défauts historiques
python3 train/train_bc.py --loss nll              # variante NLL
```

### Comment lire les sorties d'`eval_mappers.py`

- La colonne `‖v_pred−v_src‖` est LA métrique de transfert : c'est l'écart de
  vitesse d'effecteur que le robot cible produira réellement.
- La zone `centre (r<1.0)` teste le régime « bras plié » (symptôme d'angle),
  la zone `bord (r>2.5)` le régime « bras tendu » (symptôme de norme).
- La `MSE artic.` du baseline Jacobien est *volontairement* non nulle : il
  choisit une autre solution de redondance que le dataset (norme minimale),
  tout en reproduisant mieux la vitesse d'effecteur. C'est la preuve concrète
  que la MSE articulaire n'est pas le bon objectif.

---

## 6. Ce qui a volontairement été conservé

- Les anciens modèles (`run_01`, `run_02`) et leurs classes : rechargeables
  pour comparaison (`--mapper legacy`, `--run-states run_01`).
- Le format des fichiers de trajectoires : lu tel quel, corrections faites au
  chargement (pas de réécriture des données).
- `direct_method_2/` intact, les deux anciens scripts BC intacts.
- Le format d'état 6D/8D des state mappers et les noms de fichiers modèles :
  compatibilité avec les scripts existants.

## 7. Résultats mesurés et prochaines étapes

### 7.1 Action mappers (validation offline)

Voir le tableau du §2 et `direct_method/runs/eval/metrics.csv`. Les graphiques
avant/après sont dans `direct_method/runs/eval/` :
`action_mapper_legacy_2to3_summary.png` (le symptôme : 28° d'angle près du
centre, 0,2 m/s de norme partout) contre
`action_mapper_{jacobian,conditioned}_2to3_summary.png` (erreurs résiduelles
< 1° et < 0,005 m/s sur toute la plage de rayons).

### 7.2 State mappers (validation offline, médiane/p90)

| Run | Direction | ‖FK(θ̂) − eff_src‖ (m) | ‖v_pred − v_src‖ (m/s) |
|---|---|---|---|
| run_01 (legacy) | 2→3 | 0,79 / 2,64 | 2,20 / 14,7 |
| run_01 (legacy) | 3→2 | 1,24 / 3,95 | 1,15 / 4,67 |
| **run_03_kin** | 2→3 | **0,011 / 0,059** | **0,033 / 0,080** |
| **run_03_kin** | 3→2 | **0,016 / 0,102** | **0,049 / 0,111** |

Les chiffres catastrophiques de run_01 s'expliquent : il a été entraîné sur
les features effecteur en encodage fenêtre (§1.3.a) et est ici évalué sur le
format de production (eff/3, celui que les envs fournissent au transfert).
C'est le décalage qu'il subissait réellement dans
`transfer_pushball_3to2dof.py`. run_03_kin place l'effecteur reconstruit à
~1 cm et sa vitesse à ~0,04 m/s de la référence.

### 7.3 Transfert push-ball : état et étape manquante

Le script corrigé tourne de bout en bout avec les trois mappers. MAIS sur ce
PC, `data/models/ppo_pushball_2dof_2/` ne contient pas `vec_normalize.pkl`
(copie incomplète : le script d'entraînement `agents/train/train_pushball_2dof.py`
le sauvegarde normalement à côté de `best_model.zip`). Sans ces statistiques,
la politique reçoit des obs non normalisées et échoue — quel que soit le
mapper (0 % observé, attendu). **La comparaison de taux de réussite en rollout
n'est donc pas encore significative.**

Prochaines étapes :

1. Ré-entraîner la politique PPO push-ball 2DoF sur ce PC
   (`python3 -m agents.train.train_pushball_2dof`), ce qui régénérera
   `best_model.zip` **et** `vec_normalize.pkl`, puis lancer la comparaison :
   `python3 -m direct_method.transfer_pushball_3to2dof --mapper {legacy,jacobian,conditioned} --seed 0 --episodes 100`
   (avec `--model-path`/`--vecnorm-path` pointant sur le nouveau run).
2. Si le mapper conditionné s'avère moins bon que le baseline en rollout,
   utiliser le baseline en production et réserver le réseau au choix de
   posture (résiduel dans le null-space) — architecture D de
   `AVIS_TECHNIQUE_MAPPERS.md`. Vu §7.1 (le conditionné est plus robuste aux
   singularités, le baseline meilleur en médiane), un hybride est plausible.
3. Régénérer à terme le dataset de correspondance avec des vitesses cohérentes
   (dθ3 = J3⁺ J2 dθ2 au moment de la génération) pour que la MSE articulaire
   redevienne un signal propre.
