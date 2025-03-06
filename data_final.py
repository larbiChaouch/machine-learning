import os
import pandas as pd

# Définition des chemins des fichiers d'entrée
input_dir = "Datatest/Tache2/Partie2"
polluters_file = os.path.join(input_dir, "polluters_features_preprocessed.csv")
legitimate_file = os.path.join(input_dir, "legitimate_features_preprocessed.csv")

# Chargement des fichiers
polluters_df = pd.read_csv(polluters_file, sep=',', encoding='utf-8')
legitimate_df = pd.read_csv(legitimate_file, sep=',', encoding='utf-8')

# Ajout de la colonne "Classe"
polluters_df["Classe"] = 1  # Pollueurs
legitimate_df["Classe"] = 0  # Utilisateurs légitimes

# Fusion des deux DataFrames
final_df = pd.concat([polluters_df, legitimate_df], ignore_index=True)

# Tri par UserId en ordre croissant
final_df = final_df.sort_values(by=["UserId"]).reset_index(drop=True)

# Définition du dossier de sortie
output_dir = "Datatest/Tache2_donnees_final"

# Création du dossier si nécessaire
os.makedirs(output_dir, exist_ok=True)

# Définition du nom du fichier final
final_filename = os.path.join(output_dir, "data_final.csv")

# Enregistrement du fichier final
final_df.to_csv(final_filename, index=False, sep=',', encoding='utf-8')

print(f"Fichier final enregistré dans : {final_filename}")
