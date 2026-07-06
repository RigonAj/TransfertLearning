# Avis technique sur les mappers robot-robot

Auteur: Codex, assistant IA base sur GPT-5.

Date: 2026-07-06.

Ce document est mon avis technique, apres lecture du code local dans
`robot-robot/direct_method`, `robot-robot/direct_method_2`, `robot-robot/envs`,
du rapport/projet fourni dans le depot, et apres une recherche bibliographique
rapide sur le transfert inter-robots, les espaces d'action heterogenes et la
cinematique de manipulateurs redondants.

## 1. Definition du probleme

Le projet cherche a transferer une politique PPO entre deux bras planaires:

- un bras 2DoF, action 2D, observation bras 6D;
- un bras 3DoF, action 3D, observation bras 8D;
- meme espace de travail nominal: portee maximale `3.0`;
- cinematique pure, sans dynamique de masse/inertie.

La methode directe actuelle apprend deux familles de mappers:

- `state_mapper`: transforme un etat bras 2DoF en etat bras 3DoF, et inversement;
- `action_mapper`: transforme une action/vitesse articulaire 2DoF en action/vitesse
  articulaire 3DoF, et inversement.

Les fichiers de trajectoires utilises par `direct_method` ont ce format:

- 2DoF: `[theta1, theta2, dtheta1, dtheta2, eff_x, eff_y]`;
- 3DoF: `[theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, eff_x, eff_y]`.

Le point central: la quantite vraiment commune aux deux robots n'est pas la
vitesse articulaire `dtheta`, mais la position et la vitesse cartesiennes de
l'effecteur. Pour chaque robot:

```text
position effecteur = FK(theta)
vitesse effecteur  = J(theta) @ dtheta
```

Donc une action n'a pas de signification complete si elle est separee de l'etat
du bras. La meme `dtheta` peut produire des vitesses d'effecteur differentes selon
les angles articulaires, car le Jacobien depend de `theta`.

## 2. Pourquoi le state_mapper marche mieux que l'action_mapper

Mon interpretation est la suivante: le `state_mapper` a des ancres geometriques.
Il voit les angles, les vitesses et la position de l'effecteur. Meme si le mapping
2DoF <-> 3DoF est non trivial, il apprend une correspondance dans un espace qui
contient deja l'objectif commun: `(eff_x, eff_y)`.

L'`action_mapper` actuel de `direct_method/train_actions_mapper.py` apprend
principalement:

```text
[dtheta1, dtheta2] -> [dtheta1, dtheta2, dtheta3]
```

ou l'inverse. Il ne connait pas:

- les angles du robot source;
- les angles du robot cible;
- la position de l'effecteur;
- le Jacobien local;
- la zone de singularite;
- la posture redondante choisie par le 3DoF.

Cela veut dire qu'il essaye d'apprendre un mapping statique entre des commandes
articulaires alors que le probleme reel est un mapping local et conditionne:

```text
trouver dtheta_target tel que
J_target(theta_target) @ dtheta_target
    ~= J_source(theta_source) @ dtheta_source
```

Ce probleme est sous-determine en 2DoF -> 3DoF: il existe souvent plusieurs
`dtheta_3dof` qui donnent la meme vitesse d'effecteur 2D. Sans critere
supplementaire, le reseau doit deviner quelle solution de redondance choisir.

## 3. Lecture du symptome observe

Symptome decrit:

- quand l'effecteur est loin de `(0, 0)` avec le bras presque tendu, les vitesses
  predites sont differentes mais alignees;
- quand l'effecteur se rapproche de `(0, 0)` avec le bras plie, l'angle entre les
  vitesses predites augmente, mais la norme semble proche.

Mon avis: ce symptome correspond bien a une erreur de formulation du mapping, pas
seulement a un probleme d'hyperparametres.

Bras presque tendu:

- le robot est proche d'une configuration singuliere ou mal conditionnee;
- certaines directions de mouvement sont fortement contraintes;
- le mapping peut garder la direction globale mais se tromper sur la norme car le
  gain du Jacobien varie fortement.

Bras plie proche du centre:

- plusieurs configurations peuvent donner des positions proches;
- les directions locales donnees par les colonnes du Jacobien changent beaucoup;
- une MSE sur `dtheta` peut produire une norme raisonnable tout en donnant une
  direction cartesienne fausse.

Donc le probleme principal n'est pas "le MLP ne generalise pas assez". C'est:

```text
la loss et les entrees du modele ne contraignent pas directement la vitesse
cartesienne de l'effecteur.
```

## 4. Ce que disent les papiers consultes

Sources principales consultees:

1. Abhishek Gupta, Coline Devin, YuXuan Liu, Pieter Abbeel, Sergey Levine,
   "Learning Invariant Feature Spaces to Transfer Skills with Reinforcement
   Learning", 2017.
   https://arxiv.org/abs/1703.02949

   Ce papier traite explicitement de transfert entre agents de morphologies
   differentes. L'idee utile ici est de ne pas forcer une correspondance brute de
   tous les etats/actions, mais d'apprendre ou d'utiliser un espace commun
   invariant lie a la tache. Pour nos bras, l'espace commun naturel est l'espace
   de l'effecteur: position et vitesse cartesiennes.

2. Coline Devin, Abhishek Gupta, Trevor Darrell, Pieter Abbeel, Sergey Levine,
   "Learning Modular Neural Network Policies for Multi-Task and Multi-Robot
   Transfer", 2016.
   https://arxiv.org/abs/1609.07088

   Ce papier separe les composants "robot-specific" et "task-specific". Cela
   renforce l'idee que le module lie au robot doit gerer la cinematique et la
   dimension d'action, tandis que la tache doit rester dans une representation
   commune.

3. Mohamed K. Helwa, Angela P. Schoellig,
   "Multi-Robot Transfer Learning: A Dynamical System Perspective", 2017.
   https://arxiv.org/abs/1707.08689

   Leur conclusion importante est qu'un transfer map optimal n'est pas forcement
   un mapping statique simple: il depend de regressors pertinents. Dans notre cas,
   les regressors evidents sont les angles, les vitesses et possiblement
   l'historique local.

4. Nathan Beck, Abhiramon Rajasekharan, Hieu Tran,
   "Transfer Reinforcement Learning for Differing Action Spaces via Q-Network
   Representations", 2022.
   https://arxiv.org/abs/2202.02442

   Ce travail confirme que les espaces d'action differents sont un probleme
   specifique du transfert RL, pas un detail d'implementation. Un mapping d'action
   heterogene demande une structure ou un signal supplementaire.

5. Kavinayan P. Sivakumar et al.,
   "Transfer Reinforcement Learning in Heterogeneous Action Spaces using Subgoal
   Mapping", 2024.
   https://arxiv.org/abs/2410.14484

   Ce papier explore une approche par sous-objectifs plutot que par imitation
   brute des actions quand les espaces d'action different. C'est pertinent ici:
   pour `push-ball`, il peut etre plus stable de transferer un objectif local en
   espace effecteur qu'une vitesse articulaire source.

6. Jacket Demby's, Jeffrey Uhlmann, Guilherme N. DeSouza,
   "Achieving Unit-Consistent Pseudo-Inverse-based Path-Planning for Redundant
   Incommensurate Robotic Manipulators", 2023.
   https://arxiv.org/abs/2308.02964

   Ce papier rappelle que les pseudo-inverses sur manipulateurs redondants doivent
   etre utilisees avec prudence, notamment sur les questions d'unites, de bruit et
   de conditionnement. Pour notre cas simple 2D, la pseudo-inverse reste un tres
   bon baseline, mais il faut gerer les singularites.

## 5. Mon avis

Je pense qu'il ne faut pas essayer de sauver l'`action_mapper` actuel en ajoutant
seulement des couches, des epoques ou du dropout. Cela peut ameliorer la MSE sur
les trajectoires, mais cela ne garantit pas que l'effecteur ait la bonne vitesse
dans l'environnement cible.

La bonne direction est de reformuler l'action mapping autour de l'espace tache:

```text
source action -> source end-effector velocity -> target action
```

Le mapping d'action doit etre conditionne par l'etat source et l'etat cible. Dans
le transfert 2DoF -> 3DoF, la signature minimale devrait ressembler a:

```text
Ta_2to3(a_2, state_2_mapped, state_3_current) -> a_3
```

ou, au niveau cinematique:

```text
Ta_2to3(dtheta_2, theta_2, theta_3) -> dtheta_3
```

Pendant le transfert `push-ball`, le flux correct serait:

```text
state_3_current
  -> state_mapper_3to2
  -> state_2_for_policy
  -> PPO_2DoF donne action_2
  -> action_mapper(action_2, state_2_for_policy, state_3_current)
  -> action_3
```

L'action mapper ne doit donc pas seulement connaitre l'action produite par PPO;
il doit aussi connaitre les deux configurations locales qui definissent les
Jacobiens.

## 6. Plan d'action recommande

### Etape 1: ajouter des metriques avant de changer les modeles

Avant de re-entrainer, mesurer sur validation:

- erreur MSE articulaire;
- erreur de position effecteur:
  `||FK_target(theta_target_pred) - FK_source(theta_source)||`;
- erreur vectorielle de vitesse:
  `||J_target(theta_target) @ dtheta_target_pred - J_source(theta_source) @ dtheta_source||`;
- erreur de norme de vitesse;
- erreur d'angle entre vitesses;
- resultats par zone: bras tendu, bras plie, proche de `(0, 0)`, proche de la
  portee maximale.

Sans ces metriques, on risque d'optimiser une loss qui ne correspond pas au
probleme reel.

### Etape 2: creer un baseline analytique Jacobien

Je commencerais par un baseline sans reseau pour l'action mapping.

Pour 2DoF -> 3DoF:

```text
v_eff_2 = J2(theta_2) @ dtheta_2
dtheta_3 = pinv_damped(J3(theta_3)) @ v_eff_2
```

Pour 3DoF -> 2DoF:

```text
v_eff_3 = J3(theta_3) @ dtheta_3
dtheta_2 = pinv_damped(J2(theta_2)) @ v_eff_3
```

Utiliser une pseudo-inverse amortie plutot qu'une inverse brute:

```text
J_dls_pinv = J.T @ inv(J @ J.T + lambda^2 I)
```

Pourquoi cette etape est prioritaire:

- elle teste si le probleme est bien cinematique;
- elle donne un niveau de performance minimal attendu;
- elle evite que le reseau apprenne une relation que la cinematique donne deja;
- elle gere naturellement le fait que l'espace commun soit la vitesse de
  l'effecteur.

Pour le 3DoF, ajouter ensuite un terme de null-space pour choisir une posture:

```text
dtheta_3 = J3_pinv @ v_eff_2 + (I - J3_pinv @ J3) @ z
```

`z` peut servir a minimiser la norme articulaire, rester proche de la posture
courante, eviter les limites ou eviter les singularites.

### Etape 3: entrainer un action_mapper conditionne

Si le baseline analytique est meilleur que le MLP actuel, entrainer ensuite un
mapper conditionne:

```text
input  = [dtheta_source, theta_source, theta_target, eff_source, eff_target]
output = dtheta_target
```

Version plus compacte:

```text
input  = [dtheta_source, theta_source, theta_target]
output = dtheta_target
```

Loss recommandee:

```text
loss =
    w_joint * MSE(dtheta_target_pred, dtheta_target_dataset)
  + w_v     * Huber(v_eff_target_pred - v_eff_source)
  + w_speed * abs(||v_eff_target_pred|| - ||v_eff_source||)
  + w_angle * (1 - cosine_similarity(v_eff_target_pred, v_eff_source))
  + w_reg   * ||dtheta_target_pred||^2
```

Avec:

```text
v_eff_source      = J_source(theta_source) @ dtheta_source
v_eff_target_pred = J_target(theta_target) @ dtheta_target_pred
```

Je mettrais plus de poids sur la loss cartesienne que sur la MSE articulaire. La
MSE articulaire n'est qu'un regularisateur vers les demonstrations, pas l'objectif
principal.

### Etape 4: corriger aussi le state_mapper avec des losses cinematiques

Le `state_mapper` semble meilleur, mais il doit aussi etre contraint explicitement
en espace tache.

Ajouter au state mapper:

```text
loss_state =
    MSE(state_pred, state_target)
  + w_fk * ||FK(theta_pred) - eff_source_or_target||^2
  + w_vel * ||J(theta_pred) @ dtheta_pred - v_eff_source||^2
  + w_cycle * ||state_back - state_input||^2
```

Il faut faire attention aux angles pres de `-pi/pi`. Une representation
`sin(theta), cos(theta)` serait plus stable qu'un angle normalise brut, surtout si
les trajectoires passent pres de la discontinuite.

### Etape 5: revoir l'alignement des trajectoires

Le rapport mentionne l'alignement temporel et l'idee de correspondance de
trajectoires. Or un alignement par index peut etre fragile si les deux robots
font la meme tache a des vitesses differentes.

Je testerais:

- alignement par distance dans l'espace effecteur;
- Dynamic Time Warping sur `(eff_x, eff_y, v_eff_x, v_eff_y)`;
- filtrage ou ponderation plus forte des zones rares: bras tendu, proche centre,
  vitesses faibles.

Le but n'est pas seulement d'avoir des paires au meme timestep, mais des paires
qui representent le meme etat de tache.

### Etape 6: valider en rollout, pas seulement sur dataset

Une bonne validation doit inclure:

- test offline sur trajectoires;
- test rollout dans `push-ball`;
- comparaison contre:
  1. action mapper actuel;
  2. baseline Jacobien;
  3. MLP conditionne;
  4. hybride Jacobien + residual appris.

Les metriques finales importantes:

- taux de succes `push-ball`;
- distance finale balle-cible;
- distance effecteur-balle;
- erreur moyenne de vitesse effecteur;
- stabilite des actions, saturation, oscillations.

## 7. Solutions potentielles a explorer

### Solution A: baseline analytique pur

Remplacer temporairement l'`action_mapper` par:

```text
dtheta_target = damped_pinv(J_target(theta_target)) @
                (J_source(theta_source) @ dtheta_source)
```

Avantage: rapide, interpretable, tres bon diagnostic.

Limite: ne choisit pas toujours la meilleure posture redondante et peut etre
fragile pres des singularites si l'amortissement est mal regle.

### Solution B: analytique + null-space

Ajouter un terme null-space pour le 3DoF:

```text
dtheta_target = J_pinv @ v_source + N @ z
N = I - J_pinv @ J
```

Choix possibles pour `z`:

- minimiser la norme des vitesses articulaires;
- garder `theta3` proche d'une posture naturelle;
- maximiser la manipulabilite;
- rester proche de la trajectoire 3DoF alignee.

Avantage: exploite proprement la redondance 3DoF.

Limite: demande de choisir une preference de posture.

### Solution C: MLP conditionne avec loss cartesienne

Garder un reseau, mais lui donner les variables necessaires et l'entrainer avec
des losses sur `v_eff`.

Avantage: compatible avec l'approche actuelle.

Limite: demande plus de code de validation et peut apprendre des artefacts si les
donnees sont mal alignees.

### Solution D: reseau residual sur baseline Jacobien

Faire predire au reseau une correction:

```text
dtheta_target = dtheta_jacobian + residual_network(...)
```

Avantage: le baseline garantit une base cinematique correcte, le reseau corrige
les biais de donnees ou choisit la redondance.

Limite: il faut limiter le residual pour ne pas casser la garantie cinematique.

### Solution E: transferer des sous-objectifs plutot que des actions

Pour `push-ball`, au lieu de transferer directement l'action articulaire, produire
un sous-objectif en espace effecteur:

```text
desired_effector_velocity
ou
desired_next_effector_position
```

Puis chaque robot convertit ce sous-objectif en action articulaire avec son propre
controleur IK/Jacobien.

Avantage: c'est probablement la representation la plus naturelle pour transferer
entre morphologies.

Limite: cela s'eloigne du "full-policy transfer" strict, car on insere un
controleur intermediaire.

## 8. Priorite concrete pour ce depot

Je ferais dans cet ordre:

1. Ajouter un module `direct_method/kinematics.py` avec FK/Jacobiens 2DoF et 3DoF
   en NumPy et, si possible, en PyTorch differentiable.
2. Ajouter un script d'evaluation offline qui calcule les erreurs effecteur
   position/vitesse pour les modeles existants.
3. Implementer le baseline `damped_pinv` pour remplacer l'action mapper pendant
   un test `push-ball`.
4. Corriger les dimensions dans `transfer_pushball_3to2dof.py`: le state mapper
   actuel est 6D <-> 8D, mais ce script semble utiliser 4D <-> 6D.
5. Reprendre `direct_method_2` proprement ou le fusionner dans `direct_method`:
   il contient la bonne intuition, conditionner l'action par `q`, mais plusieurs
   scripts semblent desynchronises.
6. Entrainer un nouveau `ActionMapperConditionedMLP` avec la loss cartesienne.
7. Comparer sur les memes seeds et les memes episodes.

## 9. Conclusion

Mon avis final: le bon objectif n'est pas de faire correspondre les actions
articulaires, mais de faire correspondre l'effet de ces actions dans l'espace de
la tache.

Pour ce projet, cela signifie:

```text
les etats doivent matcher par FK(theta),
les actions doivent matcher par J(theta) @ dtheta.
```

Tant que l'`action_mapper` ne voit pas `theta_source` et `theta_target`, et tant
que sa loss ne penalise pas directement la vitesse cartesienne de l'effecteur, il
peut obtenir une MSE correcte tout en produisant une mauvaise action dans
l'environnement.

La meilleure prochaine experience est donc un baseline Jacobien/pseudo-inverse
amortie. Si ce baseline regle une grande partie du probleme, alors le reseau doit
devenir soit un residual autour de ce baseline, soit un module de choix de
redondance/posture, pas un remplacant aveugle de la cinematique.
