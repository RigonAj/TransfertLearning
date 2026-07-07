# Modifications du 2026-07-07 — state mapper analytique, passthrough effecteur, visualiseur corrigé

Suite de `CHANGEMENTS_ET_RAISONNEMENT.md` (2026-07-06). Ce document explique le
paradoxe « run_03_kin semble pire que run_01 en visualisation mais gagne au
transfert », les modifications apportées, et les résultats mesurés.

---

## 1. Le paradoxe visualisation / rollout — résolu

### 1.1 Le constat

Au transfert push-ball (`ppo_pushball_2dof_3`, 100 épisodes, seed 0) :

| State mapper | legacy | jacobian | conditioned |
|---|---|---|---|
| run_01 | 0 % | 0 % | 0 % |
| run_03_kin | 6 % | 5 % | 14 % |

… mais dans `visualize_state_mapper.py`, run_03_kin *semblait* moins bon que
run_01.

### 1.2 La cause : le visualiseur n'avait pas été corrigé

`visualize_state_mapper.py` normalisait les colonnes effecteur du fichier de
trajectoires **telles quelles** (`raw[:, 4:6] / MAX_REACH`). Or ces colonnes
sont en encodage fenêtre `(eff + 5) / 10`, pas en mètres (bug §1.3.a du
document précédent). Le visualiseur produisait donc des features effecteur
dans [0,09 ; 0,27] :

- **run_01** a été entraîné sur exactement cette distribution buguée → il
  paraissait bon *dans le visualiseur* ;
- **run_03_kin** a été entraîné sur les features corrigées `FK(θ)/3`
  (plage [−1 ; 1]) → il recevait des entrées hors distribution → il paraissait
  mauvais.

Au transfert, les environnements fournissent `eff/3` : c'est run_03_kin qui
est en distribution (erreur effecteur ~1,6 cm) et run_01 hors distribution
(erreur 0,8–1,2 m, cf. §7.2 du document précédent) — d'où le 0 %.

**La visualisation mentait, pas le rollout.** Le script est corrigé (§3.3).

### 1.3 Le plafond mesuré

Le PPO 2DoF `ppo_pushball_2dof_3` (budget test 15 M) réussit **72 %** en natif
sur son propre environnement (100 épisodes, seed 0, max 300 pas, mêmes
conditions que le transfert). Tout taux de transfert se lit relativement à ce
plafond.

### 1.4 Le goulot d'étranglement identifié

La balle n'interagit qu'avec l'**effecteur** (`env_pushball_3dof.py`, seuil de
contact `eff_radius + ball_radius = 0,2 m`) : un mapping parfait donne une
physique de tâche strictement équivalente. Or l'erreur de position effecteur
de run_03_kin (direction 3→2) atteint **10 cm au p90** — la moitié du seuil de
contact — alors que balle et cible sont transmises exactes. La politique
visait donc avec un effecteur perçu décalé de jusqu'à 10 cm. C'est le premier
poste d'erreur, avant l'action mapper.

---

## 2. L'idée : remplacer le state mapper 3→2 par de la cinématique exacte

Même raisonnement que pour le baseline Jacobien de l'action mapper : la
correspondance d'états 3DoF → 2DoF est **entièrement déterminée par la
géométrie**, il n'y a rien à apprendre.

```
eff      = FK3(θ3)                    exacte, partagée par les deux robots
θ2       = IK2(eff)                   forme fermée (l1 = l2 = 1,5), 2 branches
dθ2      = J2(θ2)⁺ @ (J3(θ3) @ dθ3)   pseudo-inverse amortie
```

L'IK 2DoF a deux solutions (coude haut/bas). Le mapper est *stateful* :

- premier pas d'un épisode : branche dont la courbure correspond à celle du
  bras 3DoF (signe de θ2+θ3) ;
- pas suivants : branche la plus proche de la posture précédente → continuité
  temporelle des observations (pas de saut de posture/vitesse pour la
  politique).

Erreurs mesurées (auto-test, 50 trajectoires lisses × 100 pas) :

| Métrique | Mapper appris run_03_kin (médiane/p90) | Analytique |
|---|---|---|
| ‖FK2(θ2) − FK3(θ3)‖ | 0,016 / 0,102 m | **~10⁻⁶ m** (limite float32) |
| ‖v2 − v3‖ | 0,049 / 0,111 m/s | **0,003 m/s** (médiane) |

---

## 3. Changements fichier par fichier

### 3.1 `robot-robot/direct_method/state_mapper_analytic.py` (nouveau)

`AnalyticStateMapper3to2(lam=0.05)` : entrée = bloc bras 3DoF normalisé (8D),
sortie = bloc bras 2DoF normalisé (6D), mêmes conventions que les envs
(θ/π, dθ/ω_max, eff/max_reach). `reset()` à chaque début d'épisode
(réinitialise le suivi de branche IK). Auto-test intégré :

```bash
cd robot-robot && python3 -m direct_method.state_mapper_analytic
```

Vérifie : position exacte, erreur de vitesse < 0,02 m/s hors singularités du
2DoF, aucun flip de branche (les sauts résiduels près de r ≈ 0 sont
physiquement obligatoires : θ1 du 2DoF bascule quand l'effecteur traverse
l'origine).

### 3.2 `robot-robot/direct_method/transfer_pushball_3to2dof.py`

1. **`--run-state analytic` (nouveau défaut)** : utilise le mapper analytique.
   Les valeurs `run_03_kin`, `run_01`… restent acceptées pour comparaison.
2. **Passthrough effecteur** (state mappers appris uniquement) : les
   composantes effecteur de l'état mappé sont remplacées par l'effecteur 3DoF
   réel — identique pour les deux robots, même convention `eff/3`. Supprime
   directement les 10 cm d'erreur p90 sur la composante critique (§1.4).
   Désactivable par `--no-eff-passthrough` (pour reproduire les anciens
   chiffres en A/B).
3. **Défauts `--model-path`/`--vecnorm-path`** : `ppo_pushball_2dof_1` (qui
   n'existe pas sur ce PC) → `ppo_pushball_2dof_3`. La commande nue
   `python3 -m direct_method.transfer_pushball_3to2dof --mapper X --seed 0`
   fonctionne désormais telle quelle.
4. Le state mapper est maintenant derrière une interface `(fn, reset_fn)` et
   `reset_fn()` est appelé à chaque `env.reset()`.

### 3.3 `robot-robot/direct_method/visualize_state_mapper.py`

- `normalize_2dof` recalcule les features effecteur par `FK(θ)/max_reach`
  comme `dataset.py` (correction du bug §1.2 : l'ancienne version divisait
  l'encodage fenêtre par 3). Vérifié : la nouvelle normalisation reproduit
  `FK(θ)/3` à 0 près, plage [−0,73 ; 1,0] contre [0,09 ; 0,27] avant.
- Défaut `--run-id` : `run_01` → `run_03_kin`.

Avec cette correction, run_03_kin paraît (correctement) bon et run_01
(correctement) mauvais dans le visualiseur — cohérent avec le rollout.

### 3.4 `robot-robot/agents/train/train_pushball_2dof.py`

- `TOTAL_TIMESTEPS` : 15 M → **40 M** (demande utilisateur ; le run 15 M
  plafonnait à 72 % en natif). Compter ~2,7× la durée du run 15 M.
- `run_id` : 3 → **4** (écrit dans `data/models/ppo_pushball_2dof_4/`, sans
  écraser le run 3).

---

## 4. Résultats mesurés (100 épisodes, seed 0, PPO `ppo_pushball_2dof_3`)

| State mapper | Action mapper | Taux de réussite |
|---|---|---|
| run_03_kin, sans passthrough (avant) | legacy | 6 % |
| run_03_kin, sans passthrough (avant) | jacobian | 5 % |
| run_03_kin, sans passthrough (avant) | conditioned | 14 % |
| run_03_kin + passthrough | jacobian | 4 % |
| run_03_kin + passthrough | conditioned | **17 %** |
| **analytic** | jacobian | **20 %** |
| **analytic** | conditioned | 10 % |
| *(plafond : PPO 2DoF natif)* | — | *72 %* |

Lecture :

- **meilleure config actuelle : `--run-state analytic --mapper jacobian`
  (20 %, 4× l'ancien score du jacobian)**. Le baseline Jacobien calcule
  `J2(θ2)` à partir du θ2 mappé : avec l'IK exacte, ce Jacobien est enfin
  cohérent avec l'effecteur réel — c'est pour ça qu'il profite le plus ;
- le mapper **conditioned** baisse légèrement avec l'IK analytique (14→10 %) :
  il a été entraîné sur la convention de posture du *dataset* de
  correspondance, et l'IK choisit parfois une autre branche/posture → entrées
  légèrement hors distribution pour le réseau. Avec run_03_kin (même
  convention que le dataset) + passthrough, il monte à 17 % ;
- à 100 épisodes, l'incertitude est d'environ ±4 points : 17 % et 20 % sont
  statistiquement équivalents, comparer sur 300+ épisodes ou plusieurs seeds
  avant de conclure ;
- l'écart restant vers 72 % vient du plafond de la politique (15 M de steps),
  de la perte de la pseudo-inverse amortie près des singularités du 2DoF, et
  des vitesses d'effecteur que le 2DoF virtuel ne peut pas suivre (passages
  près de r ≈ 0 et r ≈ 3).

---

## 5. Commandes à lancer (dans l'ordre)

```bash
cd ~/Documents/transfer_learning && source .venv/bin/activate && cd robot-robot

# 1. Auto-test du nouveau mapper analytique (~5 s)
python3 -m direct_method.state_mapper_analytic

# 2. Ré-entraîner le PPO 2DoF, budget 40 M  ->  data/models/ppo_pushball_2dof_4
#    (~2,7× la durée du run 15 M)
python3 -m agents.train.train_pushball_2dof

# 3. Comparaison des mappers sur le nouveau PPO (mêmes seeds)
#    ATTENTION : utiliser ppo_pushball_final.zip, PAS best_model.zip (cf. §7.1)
PPO_DIR=data/models/ppo_pushball_2dof_4
for m in legacy jacobian conditioned; do
  python3 -m direct_method.transfer_pushball_3to2dof --mapper $m --episodes 300 --seed 0 \
      --model-path $PPO_DIR/ppo_pushball_final.zip --vecnorm-path $PPO_DIR/vec_normalize.pkl
done
# variante state mapper appris + passthrough :
python3 -m direct_method.transfer_pushball_3to2dof --mapper conditioned --run-state run_03_kin \
    --episodes 300 --seed 0 --model-path $PPO_DIR/ppo_pushball_final.zip --vecnorm-path $PPO_DIR/vec_normalize.pkl
```

Depuis le ré-entraînement, ces chemins sont les défauts : les commandes
fonctionnent sans `--model-path`/`--vecnorm-path`. Le plafond natif se mesure
avec `python3 -m direct_method.eval_native_pushball` (script versionné), même
protocole que le transfert.

---

## 6. Prochaines étapes suggérées

1. **Régénérer le dataset de correspondance avec l'IK analytique** : plutôt
   que des trajectoires de reaching, échantillonner tout l'espace d'états et
   générer les paires par `θ2 = IK2(FK3(θ3))`, `dθ2 = J2⁺ J3 dθ3` (et
   `dθ3 = J3⁺ J2 dθ2` en 2→3). Données infinies, propres, couverture complète
   — puis ré-entraîner le mapper conditioned sur cette convention (il
   redeviendrait compatible avec `--run-state analytic`).
2. **Hybride action mapper** : jacobian + résiduel appris dans le null-space
   (architecture D de `AVIS_TECHNIQUE_MAPPERS.md`) — le conditioned reste
   meilleur près des singularités, le jacobian partout ailleurs.
3. Évaluer sur 300+ épisodes / plusieurs seeds avant toute conclusion fine.

---

## 7. Résultats après ré-entraînement PPO 40 M (2026-07-07, soir)

### 7.1 Piège découvert : `best_model.zip` n'est pas le meilleur modèle

Sur `ppo_pushball_2dof_4` (40 M de steps), en éval native (100 épisodes,
seed 0, déterministe) :

| Modèle | Réussite native |
|---|---|
| `best_model.zip` (snapshot au timestep 22 M) | 59 % |
| **`ppo_pushball_final.zip` (40 M)** | **99 %** |

Deux causes :

1. la sélection « best » du `EvalCallback` repose sur le **reward moyen de
   20 épisodes** — bruité, et le reward shaped (progression, proximité) ne
   coïncide pas avec le taux de succès ;
2. le `vec_normalize.pkl` sauvegardé au moment du « best » est **écrasé par la
   sauvegarde finale** (même chemin) : `best_model` (stats du 22 M) est évalué
   avec les stats du 40 M.

→ Utiliser `ppo_pushball_final.zip` (défaut du script de transfert désormais).
Pour les prochains runs : sélectionner le best sur le taux de succès et/ou
sauvegarder un pkl distinct par snapshot.

### 7.2 Transfert push-ball avec le PPO 40 M (100 épisodes, seed 0)

| State mapper | Action mapper | Taux |
|---|---|---|
| **analytic** | **jacobian** | **78 %** (74 % sur 300 ép.) |
| analytic | conditioned | 43 % |
| run_03_kin + passthrough | conditioned | 18 % |
| run_03_kin + passthrough | jacobian | 4 % |
| run_03_kin + passthrough | legacy | 3 % |
| *(plafond : PPO final natif)* | — | *99 %* |

Le pipeline entièrement analytique transfère donc **~75–80 % de la
performance de la politique source**, sans aucun mapper appris. La hiérarchie
est limpide : chaque composant appris remplacé par la cinématique exacte fait
gagner un facteur 2 à 4.

### 7.3 Réglage de l'amortissement `--lam`

Balayage sur analytic + jacobian (seed 0) :

| lam | 100 ép. | 300 ép. |
|---|---|---|
| 0,05 (ancien défaut) | 78 % | 74 % |
| **0,02 (nouveau défaut)** | 86 % | **79 %** |
| 0,01 | 83 % | — |
| 0,005 | 81 % | — |

Moins d'amortissement = meilleur suivi de la vitesse d'effecteur ; 0,02 reste
stable près des singularités. Défaut du script passé de 0,05 à 0,02.

Vérification de robustesse : seed 1, 100 épisodes, lam 0,05 → 82 %. Les
chiffres à 100 épisodes portent une incertitude d'environ ±4 points.

### 7.4 Bilan de la journée

| Étape | Taux (100 ép., seed 0) |
|---|---|
| Point de départ (PPO 15 M, run_03_kin, meilleur mapper) | 14 % |
| + state mapper analytique + action jacobian | 20 % |
| + PPO 40 M (modèle final, pas best_model) | 78 % |
| + lam 0,02 | **86 %** (79 % sur 300 ép.) |

### 7.5 Nouveaux scripts de vérification et de visualisation

- **`direct_method/eval_native_pushball.py`** : taux de réussite natif d'un
  PPO push-ball 2DoF sur son propre env (le plafond du transfert), même
  protocole que le script de transfert. Défauts = PPO final 40 M.
- **`direct_method/visualize_transfer.py`** : visualisation du transfert avec
  légende — bras 3DoF réel, bras 2DoF virtuel reconstruit par le state mapper
  (pointillés), balle, cible avec rayon de succès, et flèches de vitesse
  d'effecteur commandée (PPO, via J2) vs exécutée (3DoF, via J3), superposées
  quand le mapping est bon. Fenêtre interactive par défaut ; `--save` écrit un
  GIF par épisode dans `runs/eval/transfer_gifs/` (avec `MPLBACKEND=Agg` sans
  écran). Mêmes options de mappers que le script de transfert.
- `COMMANDES.md` §6/6bis/8-bloc C mis à jour : commandes de vérification de
  toutes les valeurs de ce document, avec les taux attendus en commentaire.

### 7.6 Ce qu'il reste à faire

1. **Court terme** : rien de bloquant — la commande nue donne la meilleure
   config (`python3 -m direct_method.transfer_pushball_3to2dof --seed 0`,
   défauts : analytic + jacobian + PPO final 40 M + lam 0,02).
2. **Combler l'écart 79 % → 99 %** : les échecs restants se concentrent sur
   les passages près des singularités du 2DoF virtuel (r ≈ 0 et r ≈ 3) où la
   vitesse d'effecteur demandée n'est pas réalisable — pistes : terme de
   null-space (`null_space_gain > 0`) pour éloigner le 3DoF de ses propres
   singularités, ou hybride jacobian + résiduel appris (architecture D).
3. **Ré-entraîner le conditioned sur un dataset IK-cohérent** (§6.1) s'il doit
   rester dans la comparaison — sinon le pipeline analytique suffit.
4. **Corriger la sélection du best model** dans
   `agents/train/train_pushball_2dof.py` (succès plutôt que reward, pkl par
   snapshot) avant le prochain gros run PPO.
5. **Direction inverse (2DoF ← politique 3DoF)** : non couverte par le script
   actuel ; l'IK analytique 2→3 demande une règle de redondance (null-space).
