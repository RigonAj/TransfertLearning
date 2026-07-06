# AGENTS.md

## Contexte du projet

Ce depot etudie le transfer learning entre bras robotiques planaires 2D ayant des
morphologies differentes. Le cas principal dans `robot-robot` est le transfert
entre un bras source 2DoF et un bras cible 3DoF, ou inversement.

Objectif general: reutiliser une politique PPO apprise sur un robot source dans
un environnement cible, sans re-entrainer toute la politique, en apprenant des
mappers entre les espaces d'etats et d'actions des deux robots.

Le rapport PDF de contexte decrit cette approche comme du "Full-Policy Transfer
via Trajectory Correspondence":

- `Ts`: mapping d'observation/etat entre l'environnement cible et l'environnement
  source.
- `Ta`: mapping d'action entre l'action produite par la politique source et
  l'action executable par le robot cible.
- Tache fondamentale: `reaching`, utilisee pour apprendre/aligner les trajectoires.
- Tache de transfert plus complexe: `push-ball`, ou le bras doit pousser une balle
  vers une cible.

## Organisation utile du depot

- `robot-robot/envs/`: environnements Gymnasium des bras 2DoF et 3DoF.
- `robot-robot/agents/train/`: entrainement PPO des politiques de base.
- `robot-robot/agents/record/`: collecte de trajectoires depuis les politiques PPO.
- `robot-robot/direct_method/`: methode directe actuelle pour entrainer les mappers.
- `robot-robot/direct_method_2/`: variante experimentale, notamment pour un
  `action_mapper` conditionne par la configuration articulaire. Ne pas supposer que
  tous les scripts y sont coherents ou executables sans correction.
- `database-robot/`: version/provenance de donnees et modeles; utile comme reference
  mais le diagnostic demande concerne surtout `robot-robot`.

## Robots et environnements

Les bras sont purement cinematiques. Il n'y a pas de dynamique de masse, inertie
ou couples moteurs. Les actions commandent des increments/vitesses articulaires.

### Bras 2DoF

Fichier principal: `robot-robot/envs/arm_2dof.py`

- Deux articulations revolutes.
- Longueurs: `l1 = 1.5`, `l2 = 1.5`.
- Portee maximale: `max_reach = 3.0`.
- Observation bras 6D normalisee:
  `[theta1/pi, theta2/pi, dtheta1/omega_max, dtheta2/omega_max, eff_x/max_reach, eff_y/max_reach]`
- Action 2D normalisee dans `[-1, 1]`.

### Bras 3DoF

Fichier principal: `robot-robot/envs/arm_3dof.py`

- Trois articulations revolutes.
- Longueurs: `l1 = 1.0`, `l2 = 1.0`, `l3 = 1.0`.
- Portee maximale: `max_reach = 3.0`.
- Observation bras 8D normalisee:
  `[theta1/pi, theta2/pi, theta3/pi, dtheta1/omega_max, dtheta2/omega_max, dtheta3/omega_max, eff_x/max_reach, eff_y/max_reach]`
- Action 3D normalisee dans `[-1, 1]`.

### Reaching

Fichiers:

- `robot-robot/envs/env_reaching_2dof.py`
- `robot-robot/envs/env_reaching_3dof.py`

La tache consiste a rapprocher l'effecteur d'une cible. L'observation complete
ajoute au bloc bras une partie tache:

- 2DoF: observation 11D = 6D bras + 5D tache.
- 3DoF: observation 13D = 8D bras + 5D tache.

La partie tache contient notamment l'erreur relative cible-effecteur, la position
de la cible et la distance normalisee.

### Push-ball

Fichiers:

- `robot-robot/envs/env_pushball_2dof.py`
- `robot-robot/envs/env_pushball_3dof.py`

La tache consiste a pousser une balle vers une cible. L'observation complete
ajoute au bloc bras les positions normalisees de la balle et de la cible:

- 2DoF: observation 10D = 6D bras + 4D tache.
- 3DoF: observation 12D = 8D bras + 4D tache.

## Termes importants

### DoF

`DoF` signifie "degree of freedom", ou degre de liberte. Ici, cela correspond au
nombre d'articulations commandees:

- 2DoF: action 2D, deux vitesses/increments articulaires.
- 3DoF: action 3D, trois vitesses/increments articulaires.

### Etat / observation

Dans ce projet, le terme `etat` est souvent utilise pour le bloc robot:

- angles articulaires `theta`;
- vitesses articulaires `dtheta`;
- position cartesienne de l'effecteur `(eff_x, eff_y)`.

Attention: les observations Gymnasium completes incluent aussi les variables de
tache, par exemple cible, balle, distances. Les mappers de `direct_method` ne
travaillent actuellement que sur le bloc bras en 6D ou 8D, pas sur toute
l'observation de `reaching` ou `push-ball`.

### Action

Une action est une commande articulaire normalisee. Dans les environnements,
elle est clippee dans `[-1, 1]`, puis transformee en variation d'angle ou vitesse
effective via `omega_max`, `dt` ou `delta_max`.

Dans les fichiers de trajectoires de `direct_method`, l'action a apprendre
correspond au bloc des vitesses articulaires:

- 2DoF: `[dtheta1, dtheta2]`;
- 3DoF: `[dtheta1, dtheta2, dtheta3]`.

### Effecteur

L'effecteur est l'extremite du bras. Sa position cartesienne est calculee par la
cinematique directe:

- 2DoF: `x = l1*cos(theta1) + l2*cos(theta1 + theta2)`,
  `y = l1*sin(theta1) + l2*sin(theta1 + theta2)`.
- 3DoF: meme principe avec un troisieme segment.

Sa vitesse cartesienne n'est pas une composante directement apprise dans
`direct_method`. Elle se calcule par le Jacobien:

`v_eff = J(theta) @ dtheta`

Ce point est central pour diagnostiquer l'`action_mapper`: la meme vitesse
articulaire `dtheta` ne donne pas la meme vitesse d'effecteur selon la
configuration `theta`.

### Jacobien

Le Jacobien `J(theta)` relie les vitesses articulaires aux vitesses cartesiennes
de l'effecteur. Il depend des angles du bras.

Consequences:

- La qualite d'un mapping d'action doit etre evaluee dans l'espace cartesien de
  l'effecteur, pas seulement par MSE sur les composantes de `dtheta`.
- Un mapper qui ne voit que l'action source ne peut pas savoir quel Jacobien
  s'applique.
- Quand le bras est presque tendu, proche d'une singularite, de petites erreurs
  articulaires peuvent changer fortement la norme ou la direction de `v_eff`.
- Quand le bras est plie et proche de `(0, 0)`, plusieurs configurations peuvent
  atteindre des positions proches mais produire des directions de vitesse
  differentes.

### State mapper

Dans `robot-robot/direct_method/train_states_mapper.py`, le `state_mapper`
apprend:

- `robot_2dofs -> robot_3dofs`: 6D vers 8D;
- `robot_3dofs -> robot_2dofs`: 8D vers 6D.

Les donnees viennent de:

- `robot-robot/direct_method/trajectories/robot_2dofs.txt`
- `robot-robot/direct_method/trajectories/robot_3dofs.txt`

Format par ligne:

- 2DoF: `[theta1, theta2, dtheta1, dtheta2, eff_x, eff_y]`
- 3DoF: `[theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, eff_x, eff_y]`

ATTENTION (verifie numeriquement le 2026-07-06):

- les colonnes `eff_x, eff_y` des fichiers sont en encodage fenetre
  `(eff + 5) / 10`, PAS en metres — `direct_method/dataset.py` les recalcule
  desormais par cinematique directe `FK(theta) / max_reach`;
- les positions effecteur des deux robots sont exactement identiques ligne a
  ligne (`||FK2 - FK3|| = 0`), mais PAS les vitesses
  (`||v2 - v3||` mediane ~0.26 m/s): les `dtheta` 3DoF du dataset ne
  reproduisent pas la vitesse d'effecteur du 2DoF. Voir
  `CHANGEMENTS_ET_RAISONNEMENT.md`.

Normalisation appliquee par `dataset.py`:

- `theta / pi`;
- `dtheta / omega_max` avec `omega_max = 2.0`;
- position effecteur recalculee `FK(theta) / max_reach` avec `max_reach = 3.0`.

Le `state_mapper` peut sembler bon parce qu'il apprend une correspondance
geometrique complete incluant la position d'effecteur.

### Action mapper

Dans `robot-robot/direct_method/train_actions_mapper.py`, l'`action_mapper`
actuel apprend seulement:

- 2D vers 3D: `[dtheta1, dtheta2] -> [dtheta1, dtheta2, dtheta3]`;
- 3D vers 2D: inverse.

Il ne recoit pas les angles articulaires `theta`, ni la position de l'effecteur,
ni le Jacobien local. C'est probablement insuffisant pour garantir que les deux
robots produisent la meme position/vitesse d'effecteur.

La variante `robot-robot/direct_method_2/` tente de conditionner l'action mapper
avec la configuration articulaire, par exemple `[dtheta_2dof, q_2dof] -> dtheta_3dof`.
Cette direction est plus pertinente, mais les scripts de visualisation/chargement
semblent partiellement desynchronises avec la signature du modele. Les verifier
avant de s'appuyer dessus.

## Probleme a investiguer

Prompt utilisateur prevu:

> Regarde ce dossier `direct_method`, les codes a l'interieur me permettent
> d'entrainer les mappers. Le `state_mapper` semble donner de bons resultats,
> mais le `action_mapper` non. Lorsque l'effecteur est loin de `(0, 0)` avec le
> bras presque tendu, les vitesses predites sont differentes mais alignees, et
> lorsque l'effecteur se rapproche de `(0, 0)` avec le bras plie, c'est la
> difference d'angle entre les vitesses predites qui augmente mais la vitesse
> semble etre la meme. Comment ameliorer la qualite des mappers et faire en
> sorte que les etats et actions, position et vitesse de l'effecteur, soient
> correctement mappes pour correspondre dans les deux environnements ?

Interpretation technique probable:

- Le `state_mapper` apprend une correspondance de positions/configurations.
- L'`action_mapper` actuel apprend une correspondance de vitesses articulaires
  hors contexte.
- Or la vitesse d'effecteur est `J(theta) @ dtheta`; elle depend donc de l'etat.
- Le mapping 2DoF -> 3DoF est redondant: plusieurs `dtheta_3dof` peuvent produire
  une meme `v_eff`. Une MSE articulaire peut penaliser une solution cinematiquement
  correcte ou favoriser une solution qui a les bonnes composantes mais le mauvais
  effet cartesien.
- Les trajectoires alignees par timestep peuvent ne pas garantir une equivalence
  locale de Jacobien ou de vitesse d'effecteur, surtout pres de singularites ou
  dans les zones repliees.

## Pistes d'amelioration a privilegier

Pour ameliorer les mappers, raisonner en espace tache et en cinematique, pas
seulement en MSE articulaire.

1. Conditionner l'`action_mapper` par l'etat.

   Au minimum, apprendre:

   - `Ta_2to3(action_2, state_2, mapped_state_3) -> action_3`, ou
   - `Ta_2to3(dtheta_2, theta_2, theta_3) -> dtheta_3`.

   Le mapper doit connaitre les angles/configurations qui determinent les
   Jacobiens source et cible.

2. Ajouter une loss cartesienne sur la vitesse de l'effecteur.

   Pour un mapping 2DoF -> 3DoF, comparer:

   - `v2 = J2(theta2) @ dtheta2`;
   - `v3 = J3(theta3_pred_or_target) @ dtheta3_pred`.

   Penaliser:

   - erreur vectorielle `||v3 - v2||`;
   - erreur de norme `abs(||v3|| - ||v2||)`;
   - erreur d'angle ou `1 - cosine_similarity(v3, v2)`.

3. Ajouter une loss cartesienne sur la position d'effecteur pour le state mapper.

   Ne pas seulement comparer les angles predits. Verifier:

   - `FK2(theta2)` contre `FK3(theta3_pred)`;
   - position source/cible dans l'espace commun;
   - coherence cycle `state -> mapped_state -> reconstructed_state`.

4. Gerer la redondance 3DoF.

   Pour 2DoF -> 3DoF, il existe plusieurs solutions. Ajouter un critere de
   regularisation:

   - norme minimale de `dtheta_3dof`;
   - continuite temporelle;
   - proximite avec la trajectoire cible si elle existe;
   - eviter les singularites;
   - favoriser une posture naturelle ou proche du `state_mapper`.

5. Envisager un mapper analytique ou hybride pour les actions.

   Option robuste:

   - calculer `v_eff_source = J_source(theta_source) @ dtheta_source`;
   - convertir vers le robot cible par pseudo-inverse:
     `dtheta_target = pinv(J_target(theta_target)) @ v_eff_source`;
   - ajouter un terme de null-space pour la redondance 3DoF;
   - utiliser le reseau seulement pour corriger ou choisir la solution redondante.

6. Verifier les donnees et les splits.

   Les fichiers de trajectoires doivent representer des etats/actions vraiment
   correspondants. Verifier:

   - alignement temporel entre `robot_2dofs.txt` et `robot_3dofs.txt`;
   - distribution des positions d'effecteur;
   - distribution des vitesses d'effecteur, pas seulement des vitesses
     articulaires;
   - couverture des zones proches de `(0, 0)`, bras tendu, bras plie, et zones
     proches de singularites.

7. Verifier les dimensions dans les scripts de transfert.

   `direct_method/transfer_pushball_3to2dof.py` semble utiliser des dimensions
   `R1_STATE_DIM = 4` et `R2_STATE_DIM = 6`, alors que le state mapper actuel est
   entraine en 6D <-> 8D pour le bloc bras complet incluant effecteur. Ne pas
   charger un modele 6D/8D avec une architecture 4D/6D.

## Points d'attention dans le code actuel

- `direct_method/dataset.py` contient les classes de dataset les plus importantes.
- `TrajectoriesTrainingDataset` utilise bien les blocs complets
  `[theta, dtheta, eff_x, eff_y]`.
- `TrajectoriesTrainingActionDataset` extrait les vitesses articulaires, pas les
  angles. C'est corrige par rapport a une erreur ancienne mentionnee dans les
  commentaires.
- `direct_method/mapper_models.py` definit des MLP simples, sans dropout effectif
  malgre le parametre `dropout`.
- `direct_method/train_states_mapper.py` redefinit localement un dataset et un
  `StateMapperMLP`, au lieu d'importer ceux de `dataset.py` et `mapper_models.py`.
  Faire attention aux divergences.
- `direct_method/visualize_velocity_mapper.py` charge un fichier
  `state_mapper_r1_to_r2.pt` mais instancie `ActionMapperMLP(STATE_DIM_R1, STATE_DIM_R2)`.
  Le nom est trompeur: ce script visualise surtout les vitesses issues du state
  mapper, pas l'action mapper autonome.
- `direct_method_2/train_actions_mapper_2.py` ecrit ses sorties dans
  `direct_method/runs/run_02_cond`, pas dans `direct_method_2/runs`.
- `direct_method_2/visualize_velocity_mapper_2.py` semble incompatible avec la
  signature actuelle de `ActionMapperMLP.forward(act_in, q_2dof)`.

## Commandes utiles

Depuis `robot-robot`, avec l'environnement Python active:

```bash
python3 -m direct_method.train_states_mapper
python3 -m direct_method.train_actions_mapper
python3 -m direct_method.visualize_state_mapper
python3 -m direct_method.visualize_velocity_mapper
```

Depuis la racine du depot, l'installation de base est:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Definition cible de "bon mapping"

Un bon mapping ne doit pas seulement minimiser une MSE normalisee entre vecteurs
articulaires. Il doit conserver les grandeurs communes aux deux robots:

- meme position d'effecteur, ou position suffisamment proche;
- meme vitesse cartesienne d'effecteur en direction et en norme;
- comportement stable dans les zones de redondance et pres des singularites;
- coherence temporelle des trajectoires;
- compatibilite avec la politique PPO source lors du transfert `push-ball`.

Pour diagnostiquer une amelioration, mesurer explicitement:

- erreur de position effecteur: `||FK_target(theta_target_pred) - FK_source(theta_source)||`;
- erreur de vitesse effecteur: `||J_target(theta_target) @ dtheta_target - J_source(theta_source) @ dtheta_source||`;
- erreur d'angle entre vitesses effecteur;
- erreur de norme de vitesse effecteur;
- taux de succes et reward en transfert `push-ball`.
