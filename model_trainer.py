from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import os
from data_loader import DataLoader

class ModelTrainer:
    def __init__(self, X_train, y_train):
        """
        Classe pour entraîner plusieurs modèles de classification.
        :param X_train: Données d'entraînement (features).
        :param y_train: Labels d'entraînement.
        """
        self.X_train = X_train
        self.y_train = y_train
        
        # Dictionnaire des modèles
        self.models = {
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
            "NaiveBayes": GaussianNB()
        }
        
        # Dictionnaire pour stocker les performances
        self.scores = {}
        
        # Dossier pour stocker les modèles entraînés
        self.save_dir = "Datatest/Tache3/Entrainement"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train_and_evaluate(self, cv=5):
        """
        Entraîne chaque modèle et effectue une validation croisée.
        Sauvegarde également les modèles entraînés.
        :param cv: Nombre de folds pour la validation croisée.
        """
        print("\nEntraînement des modèles...")
        for name, model in self.models.items():
            print(f"Entraînement du modèle {name}...")
            
            # Entraîner le modèle
            model.fit(self.X_train, self.y_train)
            
            # Validation croisée
            accuracy = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
            
            # Stocker les scores
            self.scores[name] = {
                "Mean Accuracy": np.mean(accuracy),
                "Std Dev": np.std(accuracy)
            }
            
            # Sauvegarde du modèle entraîné
            model_path = os.path.join(self.save_dir, f"{name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Modèle {name} sauvegardé sous {model_path}")
        
        print("Entraînement terminé pour tous les modèles.")
    
    def get_results(self):
        """
        Retourne les scores des modèles sous forme de dictionnaire.
        """
        return self.scores

# Chargement des données
loader = DataLoader("Datatest/Tache2_donnees_final/data_final.csv")
X_train, y_train = loader.X_train, loader.y_train

# Entraînement des modèles
trainer = ModelTrainer(X_train, y_train)
trainer.train_and_evaluate()

# Affichage des résultats
results = trainer.get_results()
print("\nRésultats des modèles :")
for model, score in results.items():
    print(f"{model}: Mean Accuracy = {score['Mean Accuracy']:.4f}, Std Dev = {score['Std Dev']:.4f}")







