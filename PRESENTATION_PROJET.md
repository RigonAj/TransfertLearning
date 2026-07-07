# Transfert de politiques entre bras robotiques de morphologies différentes

*Document de synthèse — contexte, démarche, choix techniques et résultats.
Dernière mise à jour : 7 juillet 2026.*

---

## 1. Le problème

Entraîner une politique de contrôle par apprentissage par renforcement coûte
cher : des dizaines de millions de pas de simulation pour une tâche simple.
Ce projet étudie une question naturelle : **une politique apprise sur un robot
peut-elle être réutilisée sur un autre robot, de morphologie différente, sans
tout réapprendre ?**

Le banc d'essai est volontairement épuré : deux bras robotiques planaires
purement cinématiques (pas de masses, d'inerties ni de couples — les actions
commandent directement des vitesses articulaires) :

- un bras **2 degrés de liberté** (deux segments de 1,5 m) ;
- un bras **3 degrés de liberté** (trois segments de 1 m).

Les deux ont la même portée maximale (3 m), donc le même espace de travail,
mais des espaces d'états et d'actions **incompatibles** : le 2DoF observe et
commande 2 articulations, le 3DoF en observe et commande 3. Une politique
entraînée pour l'un ne peut même pas être *exécutée* sur l'autre — les
dimensions ne correspondent pas.

Deux tâches servent de support :

- **reaching** : amener l'extrémité du bras (l'effecteur) sur une cible —
  la tâche « fondamentale », utilisée pour aligner les deux robots ;
- **push-ball** : pousser une balle jusqu'à une cible — la tâche de
  transfert, plus difficile car elle exige un contact précis (la balle n'est
  poussée que si l'effecteur l'approche à moins de 20 cm, et le succès exige
  d'amener la balle à moins de 10 cm de la cible).

L'approche étudiée, dite de « full-policy transfer », insère deux traducteurs
entre le robot cible et la politique source :

- un **state mapper** : à chaque instant, il traduit l'état du robot cible
  (par exemple le bras 3DoF) en un état équivalent du robot source, que la
  politique sait interpréter ;
- un **action mapper** : il traduit l'action produite par la politique
  (pensée pour le robot source) en une action exécutable par le robot cible.

La politique elle-même n'est jamais modifiée : elle croit contrôler son robot
d'origine, alors qu'un autre bras exécute réellement la tâche.

---

## 2. L'idée directrice : raisonner dans l'espace de la tâche

La contribution centrale du travail récent sur ce dépôt tient en une phrase :
**la grandeur commune aux deux robots n'est pas l'articulation, c'est
l'effecteur.**

Les deux bras n'ont ni le même nombre d'articulations, ni les mêmes longueurs
de segments : comparer ou faire correspondre leurs angles articulaires n'a pas
de sens physique en soi. En revanche, ils partagent le même espace de travail,
et ce qui compte pour la tâche est ce que fait l'effecteur : sa **position**
(donnée par la cinématique directe, fonction des angles) et sa **vitesse**
(donnée par le Jacobien : la vitesse cartésienne de l'effecteur est le produit
du Jacobien — une matrice qui dépend de la posture — par les vitesses
articulaires).

Cette observation a deux conséquences structurantes :

1. **Une action isolée n'a pas de signification.** La même commande de
   vitesses articulaires produit des mouvements d'effecteur complètement
   différents selon la posture du bras, puisque le Jacobien dépend des angles.
   Un traducteur d'actions qui ne voit que l'action (comme la première version
   du projet) ne peut donc pas être correct : il lui manque l'information qui
   détermine l'effet de l'action. Il apprend au mieux une relation « moyenne »
   sur les données, fausse presque partout.

2. **Les correspondances entre robots sont des objets géométriques, pas
   statistiques.** Passer d'une posture 3DoF à une posture 2DoF équivalente,
   c'est : calculer où est l'effecteur, puis résoudre la cinématique inverse
   du 2DoF (qui a une solution en forme fermée). Passer d'une vitesse
   articulaire à l'autre, c'est : convertir en vitesse d'effecteur via le
   Jacobien source, puis reconvertir via la pseudo-inverse du Jacobien cible.
   **Il n'y a, en réalité, rien à apprendre** — du moins dans ce cadre
   cinématique idéalisé.

Le projet a donc évolué d'une approche « tout appris » (des réseaux de
neurones entraînés à imiter des correspondances de trajectoires) vers une
approche « analytique d'abord » : la cinématique exacte comme référence et
comme solution de production, les réseaux étant réservés aux cas où ils
apportent quelque chose que la géométrie ne donne pas (choix de posture parmi
plusieurs solutions, robustesse près des configurations singulières).

---

## 3. Ce qui ne marchait pas, et pourquoi

Le diagnostic a mis au jour une accumulation de problèmes indépendants, chacun
suffisant à lui seul pour ruiner le transfert. Les démêler a demandé de
mesurer chaque maillon de la chaîne séparément — c'est la leçon
méthodologique principale du projet.

### 3.1 Un traducteur d'actions aveugle à la posture

L'action mapper d'origine apprenait à convertir deux vitesses articulaires en
trois, sans jamais voir les angles des bras. Le symptôme observé était
caractéristique : bras tendu (près du bord de l'espace de travail), la
direction du mouvement était à peu près bonne mais pas son amplitude ; bras
replié (près du centre), l'amplitude était plausible mais la direction fausse.
C'est exactement la signature d'un modèle qui ignore le Jacobien : près des
singularités le *gain* du Jacobien varie énormément (erreur de norme), et près
du centre ses *directions* tournent vite avec la posture (erreur d'angle).

### 3.2 Des données de correspondance incohérentes

L'analyse des fichiers de trajectoires appariées (97 435 lignes) a révélé que
les postures des deux robots étaient parfaitement alignées (même position
d'effecteur, à la précision machine), mais **pas leurs vitesses** : l'écart
médian entre les vitesses d'effecteur des deux robots était de 0,26 m/s. Les
cibles d'apprentissage étaient donc physiquement contradictoires : un réseau
parfait, entraîné à reproduire ces vitesses articulaires, aurait reproduit
cette erreur de 0,26 m/s. Et de fait, l'erreur résiduelle du mapper appris
était exactement celle du jeu de données. Moralité : une erreur quadratique
moyenne basse sur les articulations ne prouve rien si les cibles elles-mêmes
sont incohérentes en espace de tâche.

### 3.3 Un bug d'encodage des données, et le paradoxe qui en a découlé

Les colonnes « position d'effecteur » des fichiers de trajectoires n'étaient
pas en mètres mais dans un encodage d'affichage hérité de la fenêtre de rendu.
L'ancien state mapper (run_01) a été entraîné sur cette distribution erronée ;
le nouveau (run_03_kin) sur les positions recalculées par cinématique directe,
c'est-à-dire le format que les environnements fournissent réellement au moment
du transfert.

Ce bug a produit un paradoxe instructif : dans l'ancien outil de
visualisation — qui reproduisait le même encodage erroné — le vieux mapper
paraissait *meilleur* que le nouveau, alors qu'en transfert réel c'était
l'inverse (0 % de réussite contre 5–14 %). Chaque modèle était simplement
évalué dans la distribution de l'autre. La visualisation mentait, pas le
rollout. D'où une règle adoptée depuis : **tout outil d'évaluation ou de
visualisation doit utiliser exactement le pipeline de normalisation de la
production**, et les métriques de référence se mesurent en espace de tâche
(position et vitesse d'effecteur en mètres), jamais seulement en unités
normalisées internes.

### 3.4 Le « meilleur modèle » qui n'en était pas un

Lors du ré-entraînement de la politique source, le mécanisme standard de
sélection du meilleur point de contrôle (basé sur la récompense moyenne de 20
épisodes d'évaluation périodiques) a retenu un instantané réussissant 59 % de
la tâche, alors que le modèle final en réussissait 99 %. Deux causes : la
récompense façonnée (proximité, progression, alignement) n'est pas le taux de
succès, et 20 épisodes donnent une estimation très bruitée. S'y ajoutait un
défaut de sauvegarde : les statistiques de normalisation associées au
« meilleur » instantané étaient écrasées par celles de la fin d'entraînement.
Leçon générale : **sélectionner les modèles sur la métrique que l'on veut
réellement optimiser** (ici le taux de succès), avec suffisamment d'épisodes
pour que la comparaison soit significative.

### 3.5 Le clonage de comportement bridé par deux défauts classiques

La partie « base de données » du projet (apprendre une politique par imitation
de 2 000 démonstrations réussies, sans interaction avec la récompense)
plafonnait sous les 40 % alors que l'expert imité réussit 96,5 %. Les données
étaient irréprochables ; les causes étaient ailleurs :

- **sous-capacité** : le réseau étudiant utilisait la taille par défaut de la
  bibliothèque (deux couches de 64 neurones), seize fois plus petit que
  l'expert cloné (deux couches de 256). Il ne pouvait pas représenter la
  politique experte, tout en affichant une perte de validation flatteuse — une
  erreur moyenne basse n'implique pas la précision requise aux états
  critiques, comme l'instant du contact avec la balle ;
- **dérive de distribution** : défaut structurel de l'imitation pure.
  L'étudiant n'est entraîné que sur les états visités par l'expert ; à la
  première petite déviation, il se retrouve dans des situations jamais vues où
  son comportement est arbitraire, et l'erreur s'amplifie de pas en pas. Ce
  phénomène est invisible pour la perte de validation, qui est mesurée sur la
  distribution de l'expert — d'où des courbes d'apprentissage excellentes et
  un robot médiocre.

---

## 4. L'architecture retenue

### 4.1 Une cinématique de référence, unique et testée

Toutes les quantités géométriques (cinématique directe, Jacobiens,
pseudo-inverse amortie, constantes de normalisation) sont regroupées dans un
module unique, décliné en deux versions : une version différentiable
(utilisable à l'intérieur des fonctions de coût d'entraînement) et une version
numérique pour les scripts d'évaluation et de transfert. Ce module est
auto-testé contre les environnements de simulation. Avant cette
centralisation, ces formules étaient dupliquées en plusieurs endroits — toute
divergence aurait faussé l'apprentissage silencieusement.

### 4.2 La traduction d'états : cinématique inverse exacte

Dans le sens 3DoF vers 2DoF (celui du transfert étudié), la correspondance
d'états est calculée, pas apprise : position d'effecteur par cinématique
directe (exacte et identique pour les deux robots), posture 2DoF par
cinématique inverse en forme fermée, vitesses articulaires par pseudo-inverse
du Jacobien. La cinématique inverse du 2DoF admet deux solutions (« coude en
haut » ou « coude en bas ») : le traducteur choisit initialement la branche
dont la courbure correspond à celle du bras 3DoF, puis conserve d'un pas à
l'autre la branche la plus proche de la posture précédente. Cette continuité
temporelle est importante : un basculement de branche créerait dans les
observations de la politique un saut de posture et de vitesse qu'un vrai bras
ne produirait jamais.

Le state mapper appris (un perceptron multicouche entraîné avec des pertes
cinématiques) est conservé à titre de comparaison : son erreur de position
d'effecteur, environ 1,6 cm en médiane mais jusqu'à 10 cm dans les cas
difficiles, est rédhibitoire pour une tâche où le seuil de contact est de
20 cm — la politique « visait » avec un effecteur perçu au mauvais endroit.
La version analytique ramène cette erreur à zéro.

### 4.3 La traduction d'actions : pseudo-inverse amortie du Jacobien

L'action est convertie en vitesse d'effecteur via le Jacobien du robot source,
puis reconvertie en vitesses articulaires du robot cible via la pseudo-inverse
de son Jacobien. Deux raffinements :

- **l'amortissement** de la pseudo-inverse borne le gain près des
  singularités (bras tendu), où une inversion brute exploserait. Sa valeur est
  un compromis : trop d'amortissement dégrade le suivi de vitesse partout,
  trop peu rend les singularités dangereuses. La valeur retenue (0,02) a été
  choisie par balayage en conditions réelles : elle a fait gagner environ cinq
  points de taux de succès par rapport à la valeur initiale plus prudente
  (0,05) ;
- dans le sens 2DoF vers 3DoF, le problème est **redondant** : une infinité de
  vitesses articulaires 3DoF produisent la même vitesse d'effecteur. La
  pseudo-inverse choisit la solution de norme minimale, et un terme optionnel
  de « null-space » permet d'orienter la posture (par exemple vers une
  configuration neutre) sans modifier le mouvement de l'effecteur.

Un traducteur d'actions **appris et conditionné** existe également : il reçoit
l'action source *et* les deux postures, et il est entraîné avec une fonction
de coût dominée par l'erreur de vitesse d'effecteur en unités physiques
(erreur vectorielle, de norme et d'angle), la fidélité aux vitesses
articulaires du jeu de données n'étant qu'un faible régularisateur — précisément
parce que ces cibles sont incohérentes (§3.2). Les angles lui sont présentés
en sinus/cosinus plutôt qu'en valeur brute, pour éviter la discontinuité
artificielle à ±180° et parce que c'est la forme sous laquelle les angles
entrent réellement dans les Jacobiens. Hors ligne, ce réseau est légèrement
moins précis que la pseudo-inverse en médiane, mais plus robuste dans les pires
cas près des singularités — il a appris à les contourner. En rollout complet,
c'est néanmoins le pipeline entièrement analytique qui domine, notamment parce
que le réseau conditionné reste attaché à la convention de posture du jeu de
données qui l'a formé.

### 4.4 La politique source : PPO et ses pièges d'évaluation

La politique de référence est un PPO standard (réseaux 256×256, normalisation
des observations et des récompenses, taux d'apprentissage en décroissance
linéaire), entraîné 40 millions de pas — un budget choisi comme compromis
entre le budget d'essai initial (15 M, plafonnant à 72 % de réussite) et le
budget complet historique (150 M). À 40 M, la politique atteint 99 % sur sa
propre tâche. La récompense combine un terme de progression de la balle vers
la cible, un terme de proximité bras-balle, un léger coût de commande et un
bonus de succès ; c'est efficace pour l'apprentissage mais, comme vu en §3.4,
c'est une mauvaise base de sélection de modèle — le taux de succès mesuré sur
un nombre suffisant d'épisodes est désormais la métrique de référence.

### 4.5 Le clonage de comportement : capacité alignée et DAgger

Pour la partie imitation, deux corrections correspondant aux deux causes du
§3.5 :

- le réseau étudiant adopte la **même capacité que l'expert** (256×256). À lui
  seul, ce changement fait passer le taux de succès de 36,5 % à 85 % ;
- la dérive de distribution est traitée par **DAgger** (« Dataset
  Aggregation ») : après l'imitation initiale, on fait agir l'étudiant dans
  l'environnement, on demande à l'expert quelle action il aurait prise sur
  chacun des états ainsi visités — y compris les états « hors piste » où
  l'étudiant s'égare — puis on agrège ces nouvelles paires au jeu de données
  et on ré-entraîne. En dix itérations, l'étudiant apprend à se rattraper là
  où il déviait. Le meilleur modèle est sélectionné sur le taux de succès en
  rollout (sur un jeu d'épisodes fixe, pour que les itérations soient
  comparables), et non sur la perte de validation, aveugle à la dérive.

---

## 5. Résultats

Tous les chiffres sont des taux de réussite de la tâche push-ball, mesurés en
mode déterministe. À 100 épisodes, l'incertitude statistique est d'environ
±4 points ; les conclusions fines sont confirmées sur 300 épisodes ou plus.

**Transfert 3DoF ← politique 2DoF** (la politique n'a jamais vu le bras 3DoF) :

| Configuration | Réussite |
|---|---|
| Point de départ (mappers appris, politique 15 M) | 14 % |
| Mappers analytiques (états + actions), politique 15 M | 20 % |
| Mappers analytiques, politique 40 M | 78 % |
| Mappers analytiques, amortissement affiné | **86 %** (79 % sur 300 épisodes) |
| *Référence : la politique sur son propre robot 2DoF* | *99 %* |

Le transfert restitue donc environ **80 % de la performance de la politique
source**, sans réapprentissage et sans aucun mapper entraîné. L'écart restant
se concentre sur les passages près des singularités du bras 2DoF virtuel
(effecteur près du centre ou en limite de portée), où certaines vitesses
d'effecteur demandées ne sont physiquement pas réalisables par le 2DoF que la
politique croit contrôler.

**Clonage de comportement** (imitation de 2 000 démonstrations) :

| Configuration | Réussite |
|---|---|
| Réseau 64×64, imitation pure (état initial) | 36,5 % |
| Réseau 256×256, imitation pure | 85 % |
| Réseau 256×256 + DAgger | **91,2 %** (500 épisodes indépendants) |
| *Référence : l'expert imité* | *96,5 %* |

---

## 6. Enseignements généraux

1. **Chercher d'abord la structure du problème.** Dans un cadre cinématique,
   les correspondances entre robots sont de la géométrie ; un réseau de
   neurones qui essaie de la réapprendre à partir de trajectoires part avec un
   handicap (données bruitées, incohérentes, couverture partielle) pour un
   résultat au mieux équivalent. Le bon usage de l'apprentissage est de
   compléter l'analytique (choix de redondance, robustesse aux singularités),
   pas de le remplacer.

2. **Optimiser et évaluer la grandeur qui compte.** Erreur articulaire
   moyenne, perte de validation, récompense façonnée : chacune de ces métriques
   a, à un moment du projet, donné une image inversée de la réalité. Les
   métriques fiables ici sont la position et la vitesse de l'effecteur en
   unités physiques, et le taux de succès en conditions réelles.

3. **Chaque maillon se mesure séparément.** Le taux de succès final confond la
   qualité de la politique, du state mapper, de l'action mapper et de
   l'évaluation elle-même. Les progrès n'ont été possibles qu'en isolant
   chaque étage avec sa propre métrique et sa propre référence analytique.

4. **Se méfier des distributions.** La plupart des bugs rencontrés étaient des
   décalages de distribution silencieux : encodage d'entraînement différent de
   la production, statistiques de normalisation désynchronisées, états de
   rollout absents des démonstrations. Aucun ne levait d'erreur ; tous
   détruisaient la performance.

---

## 7. Perspectives

- **Combler l'écart résiduel du transfert** (79 % contre 99 %) : exploiter le
  null-space du bras 3DoF pour éviter activement les postures où le 2DoF
  virtuel devient singulier, ou apprendre un petit correcteur résiduel
  par-dessus la solution analytique.
- **Régénérer le jeu de correspondances par cinématique inverse**, avec des
  vitesses cohérentes, pour redonner une chance équitable aux mappers appris
  et notamment au réseau conditionné.
- **Le sens inverse du transfert** (exécuter une politique 3DoF sur le bras
  2DoF) : la traduction d'états 2DoF vers 3DoF est redondante et demande une
  règle de choix de posture — le cas d'usage naturel du terme de null-space.
- **Rapprocher les deux volets du projet** : le clonage de comportement peut
  servir à distiller une politique transférée (politique source + mappers) en
  une politique native du robot cible, supprimant les traducteurs à
  l'exécution.

---

## 8. Pour aller plus loin dans le dépôt

Les documents techniques détaillés, avec commandes reproductibles et valeurs
attendues : `AGENTS.md` (référence du projet et des formats),
`AVIS_TECHNIQUE_MAPPERS.md` (analyse et bibliographie),
`CHANGEMENTS_ET_RAISONNEMENT.md` (refonte des mappers, 6 juillet),
`MODIFICATIONS_2026-07-07.md` (mapper analytique, PPO 40 M, résultats de
transfert), `AMELIORATIONS_BC_2026-07-07.md` (clonage de comportement) et
`COMMANDES.md` (toutes les commandes, entraînements, évaluations et
visualisations).
