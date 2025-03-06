import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ComparaisonTache:
    def __init__(self, tache3_path, tache4_path, output_dir="Datatest/Comparaison_Tache3_Tache4"):
        """
        Classe pour comparer les performances des modèles entre la Tâche 3 (équilibrée) et la Tâche 4 (déséquilibrée).
        :param tache3_path: Chemin du fichier CSV des résultats de la Tâche 3.
        :param tache4_path: Chemin du fichier CSV des résultats de la Tâche 4.
        :param output_dir: Dossier de sauvegarde des comparaisons.
        """
        self.tache3_path = tache3_path
        self.tache4_path = tache4_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Chargement des données
        self.df_tache3, self.df_tache4 = self.load_data()

        # Fusion des résultats
        self.df_comparatif = self.merge_data()

    def load_data(self):
        """Charge et prépare les fichiers CSV des résultats des Tâches 3 et 4."""
        if not os.path.exists(self.tache3_path) or not os.path.exists(self.tache4_path):
            raise FileNotFoundError("Assurez-vous que les fichiers de résultats des Tâches 3 et 4 existent.")

        df_tache3 = pd.read_csv(self.tache3_path, sep=',', encoding='utf-8')
        df_tache4 = pd.read_csv(self.tache4_path, sep=',', encoding='utf-8')

        # Vérifier les colonnes
        print("Colonnes Tâche 3 :", df_tache3.columns.tolist())
        print("Colonnes Tâche 4 :", df_tache4.columns.tolist())

        # Renommer la colonne 'Unnamed: 0' en 'Model' si nécessaire
        df_tache3.rename(columns={"Unnamed: 0": "Model"}, inplace=True)
        df_tache4.rename(columns={"Unnamed: 0": "Model"}, inplace=True)

        return df_tache3, df_tache4

    def merge_data(self):
        """Fusionne les résultats des deux tâches pour comparer les performances."""
        df_comparatif = self.df_tache3.merge(self.df_tache4, on="Model", suffixes=("_Tache3", "_Tache4"))
        save_path = os.path.join(self.output_dir, "comparatif_T3_T4.csv")
        df_comparatif.to_csv(save_path, index=False, sep=',', encoding='utf-8')
        print(f"Fichier de comparaison sauvegardé : {save_path}")
        return df_comparatif

    def plot_comparison(self):
        """Génère des graphiques comparatifs des performances des modèles."""
        metrics = ["TP_Rate", "FP_Rate", "F1-score", "AUC"]

        for metric in metrics:
            plt.figure(figsize=(10, 5))
            df_melted = self.df_comparatif.melt(id_vars=["Model"], value_vars=[f"{metric}_Tache3", f"{metric}_Tache4"],
                                                var_name="Tâche", value_name=metric)
            df_melted["Tâche"] = df_melted["Tâche"].replace({f"{metric}_Tache3": "Équilibré", f"{metric}_Tache4": "Déséquilibré"})

            sns.barplot(data=df_melted, x="Model", y=metric, hue="Tâche", palette=["blue", "red"])
            plt.title(f"Comparaison {metric} - Tâche 3 vs Tâche 4")
            plt.xticks(rotation=45)
            plt.legend(title="Tâche")
            save_path = os.path.join(self.output_dir, f"{metric}_comparaison.png")
            plt.savefig(save_path)
            print(f"Graphique sauvegardé : {save_path}")
            plt.close()

        # Génération du graphique combiné
        self.plot_global_comparison()

    def plot_global_comparison(self):
        """Génère un graphique combiné comparant toutes les métriques."""
        plt.figure(figsize=(12, 6))
        models = self.df_comparatif["Model"]

        for metric, color, marker in zip(["TP_Rate", "FP_Rate", "F1-score", "AUC"], ["blue", "red", "green", "purple"], ["o", "s", "D", "*"]):
            scores_t3 = self.df_comparatif[f"{metric}_Tache3"]
            scores_t4 = self.df_comparatif[f"{metric}_Tache4"]

            plt.plot(models, scores_t3, marker=marker, linestyle="-", linewidth=2, markersize=8, label=f"{metric} (Équilibré)", color=color)
            plt.plot(models, scores_t4, marker=marker, linestyle="--", linewidth=2, markersize=8, label=f"{metric} (Déséquilibré)", color=color, alpha=0.6)

        plt.xlabel("Modèles")
        plt.ylabel("Score")
        plt.title("Comparaison globale des performances des modèles (Tâche 3 vs Tâche 4)")
        plt.legend()
        save_path = os.path.join(self.output_dir, "comparaison_globale.png")
        plt.savefig(save_path)
        print(f"Graphique global sauvegardé : {save_path}")
        plt.close()

    def save_comparison_table(self):
        """Génère un tableau comparatif des résultats sous format PNG."""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=self.df_comparatif.round(4).values,
                         colLabels=self.df_comparatif.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        save_path = os.path.join(self.output_dir, "comparatif_T3_T4.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Tableau comparatif sauvegardé : {save_path}")
        plt.close()

# **Exécution du script**
if __name__ == "__main__":
    tache3_csv = "Datatest/Tache3/Results/summary_table.csv"
    tache4_csv = "Datatest/Tache4/Results/model_performance.csv"

    # Création et exécution de la comparaison
    comparateur = ComparaisonTache(tache3_csv, tache4_csv)
    comparateur.plot_comparison()
    comparateur.save_comparison_table()

    print("\nComparaison des Tâches 3 et 4 terminée avec succès !")
