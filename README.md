# README – Projet INF7370 : Apprentissage Automatique

**Nom :** Larbi Chaouch  
**Cours :** INF7370 – Apprentissage machine (Hiver 2025)  
**Professeur :** Mohamed Bouguessa  
**Date de remise :** 24 février 2025 avant 09h30  

---

## 1. Description Générale

Ce projet couvre l’ensemble du flux de travail d’un projet d’apprentissage automatique :  
1. **Préparation du jeu de données** (suppression des doublons, gestion des valeurs manquantes, extraction des attributs).  
2. **Comparaison de plusieurs algorithmes** sur un jeu de données équilibré (pollueurs vs légitimes).  
3. **Évaluation des mêmes algorithmes** sur un jeu de données artificiellement déséquilibré (pollueurs = 5 % des utilisateurs légitimes).  

Les algorithmes utilisés sont :  
- **Decision Tree**  
- **Bagging**  
- **AdaBoost**  
- **Gradient Boosting**  
- **Random Forest**  
- **Naive Bayes**  

Les métriques de comparaison incluent :  
- **Taux de vrais positifs (TP Rate)**  
- **Taux de faux positifs (FP Rate)**  
- **F1-score**  
- **Aire sous la courbe ROC (AUC)**  

Le **rapport PDF** associé détaille les choix méthodologiques, l’explication de l’algorithme **GBoost** et l’analyse de l’ensemble des résultats.

---

## 2. Organisation des Fichiers & Scripts

- **Datasets/**  
  Contient les fichiers de données brutes, par exemple :  
  - `content_polluters.txt`  
  - `legitimate_users.txt`  
  - `content_polluters_tweets.txt`  
  - `legitimate_users_tweets.txt`

- **Datatest/**  
  - **Tache_2/**  
    - Fichiers de sortie pour la préparation des données (incluant `final_dataset.csv`).  
  - **Tache_3/**  
    - Fichiers de résultats et graphiques pour la comparaison sur données équilibrées.  
  - **Tache_4/**  
    - Fichiers de résultats et graphiques pour l’analyse comparative sur données déséquilibrées.

- **features_extraction.py**  
  - Lit les données brutes dans `Datasets/`.  
  - Extrait les différentes caractéristiques (ex. durée de vie du compte, ratio followings/followers, etc.).  
  - Nettoie et normalise les données (suppression de doublons, remplacement des valeurs manquantes, etc.).  
  - Génère le fichier CSV final (`final_dataset.csv`) dans `Datatest/Tache_2/`.  

- **comparison_all_algorithms.py**  
  - Charge le CSV `Datatest/Tache_2/final_dataset.csv`.  
  - Entraîne et évalue les algorithmes (Decision Tree, Bagging, etc.).  
  - Compare les performances (TP Rate, FP Rate, F1-score, AUC).  
  - Stocke les résultats et graphiques dans `Datatest/Tache_3/`.  

- **imbalanced_data_comparison.py**  
  - Crée un sous-ensemble déséquilibré (pollueurs = 5 % des utilisateurs légitimes).  
  - Répète la comparaison des algorithmes sur ce sous-ensemble.  
  - Compare les résultats (TP Rate, FP Rate, F1-score, AUC) avec ceux obtenus sur l’ensemble équilibré.  
  - Stocke les résultats et graphiques dans `Datatest/Tache_4/`.  

- **rapport.pdf**  
  - Document détaillé contenant l’analyse de l’algorithme GBoost, la méthodologie, et l’interprétation approfondie des résultats.

---

## 3. Installation & Dépendances

### Vous pouvez executer seulement `make` et il va faire l'affaire pour la section en bas.


Le code est écrit en **Python 3.8+** (recommandé). Avant d’exécuter les scripts, assurez-vous d’installer les bibliothèques suivantes :

```bash
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install scikit-plot   # Optionnel, si vous souhaitez tracer des courbes ROC via scikit-plot
```

Si besoin, vous pouvez aussi créer un environnement virtuel :
```bash
python -m venv venv ou python3 -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
deactivate                # pour quitter l'environnement

# Puis installer les dépendances
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install scikit-plot
pip install seaborn
```

## 4. Guide d’Exécution

**1. Extraction des caractéristiques et préparation du dataset**
```bash
python features_extraction.py
# Ce script produit final_dataset.csv dans le dossier Datatest/Tache_2/.
```

**2. Comparaison des algorithmes sur données équilibrées**
```bash
python comparison_all_algorithms.py
# Les résultats (tableau comparatif, graphiques) sont sauvegardés dans Datatest/Tache_3/.
```

**3. Analyse sur données déséquilibrées (5 % de pollueurs)**
```bash
python imbalanced_data_comparison.py
# Génère un sous-ensemble déséquilibré dans Datatest/Tache_4/imbalanced_dataset.csv.
# Compare les mêmes algorithmes et génère les résultats dans Datatest/Tache_4/.
```


## 5. Résultats Attendus
- **`Datatest/Tache_2/final_dataset.csv`** : Jeu de données final après nettoyage et extraction des attributs.

- **`Datatest/Tache_3/model_comparison_results.csv`** : Résultats (métriques) des modèles sur données équilibrées.

- **`Datatest/Tache_3/graphe_model_comparison.png`, `Datatest/Tache_3/model_comparison_table.png`** : Graphiques et tableau comparatifs.

- **`Datatest/Tache_4/imbalanced_dataset.csv`** : Jeu de données déséquilibré (5 % de pollueurs).

- **`Datatest/Tache_4/imbalanced_model_comparison_results.csv`** : Résultats des modèles sur données déséquilibrées.

- **`Datatest/Tache_4/comparison_*.png`**: Graphiques de comparaison (TP Rate, FP Rate, F1-score, AUC) entre équilibré et déséquilibré.

## 6. Vérification
- **Vérifiez l’existence des dossiers** 
    `Datatest/Tache_2/`, `Datatest/Tache_3/` et `  Datatest/Tache_4/
    `

- **Les scripts créent automatiquement ces dossiers si nécessaire.**

- **Les fichiers de données brutes doivent se trouver dans le dossier
    `
    Datasets/ #(ou modifiez les  chemins directement dans le code si besoin).
    `**

- **Les deux nouvelles caractéristiques (Tâche 2) peuvent être ajoutées dans la partie feature engineering du script `features_extraction.py.`**

- **Le rapport 
`Rapport TP1 INF7370-V2.pdf`
 contient les détails théoriques (GBoost), les discussions et interprétations de résultats.**



 # --------------------------------------------
 # DerniereModification
 # --------------------------------------------

 Appliquer la transformation logarithmique log(1 + x) pour les variables qui ne peuvent pas être négatives :
NumerOfFollowings
NumberOfFollowers
TweetsPerDay
URLRatio
MeanTimeBetweenTweets
MaxTimeBetweenTweets
HashtagRatio
FollowBackRatio
Utiliser uniquement Min-Max Scaling pour LengthOfScreenName, LengthOfDescriptionInUserProfile, et MentionRatio.
Le RatioFollowingFollowers pouvant être négatif (ex. : plus de followings que de followers), il peut rester en Z-Score.
Min-Max Scaling pour LengthOfScreenName, LengthOfDescriptionInUserProfile, MentionRatio.
Transformation logarithmique (log(1 + x)) pour NumerOfFollowings, NumberOfFollowers, TweetsPerDay, etc.
Z-Score uniquement sur RatioFollowingFollowers.