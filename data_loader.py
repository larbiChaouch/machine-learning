import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_path, test_size=0.2, random_state=42):
        """
        Classe pour charger et préparer les données.
        :param data_path: Chemin du fichier CSV contenant les données finales.
        :param test_size: Proportion des données utilisées pour le test.
        :param random_state: Seed pour la reproductibilité.
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        
        # Chargement des données
        self.data = self.load_data()
        
        # Séparation des features (X) et de la cible (y)
        self.X, self.y = self.split_features_target()
        
        # Division en ensembles d'entraînement et de test
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()
    
    def load_data(self):
        """Charge les données depuis un fichier CSV."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Le fichier {self.data_path} n'existe pas.")
        
        data = pd.read_csv(self.data_path, sep=',', encoding='utf-8')
        print(f"Données chargées avec succès depuis {self.data_path}")
        print(f"Nombre total d'échantillons : {data.shape[0]}")
        return data
    
    def split_features_target(self):
        """Sépare les caractéristiques (X) et la variable cible (y)."""
        X = self.data.drop(columns=['Classe'])  # Suppression de la colonne cible
        y = self.data['Classe']  # Variable cible
        print("Séparation des features et de la cible effectuée.")
        return X, y
    
    def split_train_test(self):
        """Divise les données en ensembles d'entraînement et de test."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )
        print(f"Données divisées : {self.test_size * 100}% pour le test, {100 - self.test_size * 100}% pour l'entraînement.")
        print(f"Taille d'entraînement : {X_train.shape[0]} échantillons")
        print(f"Taille de test : {X_test.shape[0]} échantillons")
        return X_train, X_test, y_train, y_test
    
#Exemple d'utilisation
loader = DataLoader("Datatest/Tache2_donnees_final/data_final.csv")
X_train, X_test, y_train, y_test = loader.X_train, loader.X_test, loader.y_train, loader.y_test
