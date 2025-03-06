import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class DataPreparation:
    def __init__(self, input_dir):
        """
        Initialise la classe en chargeant les fichiers de données.
        :param input_dir: Chemin du dossier contenant les fichiers CSV à traiter.
        """
        self.input_dir = input_dir
        self.polluters_file = os.path.join(input_dir, "polluters_features.csv")
        self.legitimate_file = os.path.join(input_dir, "legitimate_features.csv")
        
        # Chargement des fichiers
        self.polluters_df = pd.read_csv(self.polluters_file, sep=',', encoding='utf-8')
        self.legitimate_df = pd.read_csv(self.legitimate_file, sep=',', encoding='utf-8')
        
    def remove_duplicates(self):
        """
        Supprime les doublons dans les deux DataFrames.
        """
        self.polluters_df.drop_duplicates(inplace=True)
        self.legitimate_df.drop_duplicates(inplace=True)
    
    def handle_missing_values(self):
        """
        Remplace les valeurs manquantes par la médiane de chaque colonne.
        """
        for df in [self.polluters_df, self.legitimate_df]:
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
    
    def normalize_data(self):
        """
        Normalise les colonnes numériques avec Log-Transformation ou Min-Max Scaling selon les recommandations.
        """
        # Colonnes à normaliser avec Min-Max Scaling
        min_max_cols = ['LengthOfScreenName', 'LengthOfDescriptionInUserProfile', 'MentionRatio']
        
        # Colonnes à normaliser avec transformation logarithmique (car elles doivent rester positives)
        log_transform_cols = ['AccountLongevity', 'NumerOfFollowings', 'NumberOfFollowers', 
                              'TweetsPerDay', 'URLRatio', 'MeanTimeBetweenTweets', 
                              'MaxTimeBetweenTweets', 'HashtagRatio', 'FollowBackRatio']
        
        # Colonnes qui peuvent être normalisées avec Z-Score (si elles ont des valeurs négatives)
        zscore_cols = ['RatioFollowingFollowers']
        
        min_max_scaler = MinMaxScaler()
        
        for df in [self.polluters_df, self.legitimate_df]:
            df[min_max_cols] = min_max_scaler.fit_transform(df[min_max_cols])
            
            # Transformation logarithmique (log(1 + x))
            for col in log_transform_cols:
                df[col] = np.log1p(df[col])
    
    def process(self):
        """
        Effectue toutes les étapes de nettoyage et de normalisation des données.
        """
        self.remove_duplicates()
        self.handle_missing_values()
        self.normalize_data()
        print("Prétraitement terminé : doublons supprimés, valeurs manquantes traitées, données normalisées.")

# Création et exécution du prétraitement
prep = DataPreparation("Datatest/Tache2/Partie1")
prep.process()

# Définition du dossier de sortie
output_dir = "Datatest/Tache2/Partie2"

# Création du dossier si nécessaire
os.makedirs(output_dir, exist_ok=True)

# Définition des noms de fichiers
polluters_filename = os.path.join(output_dir, "polluters_features_preprocessed.csv")
legitimate_filename = os.path.join(output_dir, "legitimate_features_preprocessed.csv")

# Enregistrement des fichiers
prep.polluters_df.to_csv(polluters_filename, index=False, sep=',', encoding='utf-8')
prep.legitimate_df.to_csv(legitimate_filename, index=False, sep=',', encoding='utf-8')

print(f"Fichiers enregistrés dans {output_dir} :\n - {polluters_filename}\n - {legitimate_filename}")


