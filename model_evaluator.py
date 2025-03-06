import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
from data_loader import DataLoader

class ModelEvaluator:
    def __init__(self, trained_models):
        self.trained_models = trained_models
        self.results = {}

        # Dossier principal pour stocker les résultats
        self.save_dir = "Datatest/Tache3/Results"
        os.makedirs(self.save_dir, exist_ok=True)

        # Sous-dossiers pour chaque métrique
        self.metric_dirs = {
            "TP_Rate": os.path.join(self.save_dir, "TP_Rate"),
            "FP_Rate": os.path.join(self.save_dir, "FP_Rate"),
            "F1-score": os.path.join(self.save_dir, "F1-score"),
            "AUC": os.path.join(self.save_dir, "AUC")
        }
        for path in self.metric_dirs.values():
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def load_models(model_dir):
        """Charge les modèles sauvegardés."""
        trained_models = {}
        if not os.path.exists(model_dir):
            return trained_models
        
        for filename in os.listdir(model_dir):
            if filename.endswith(".pkl"):
                model_name = filename.replace(".pkl", "")
                model_path = os.path.join(model_dir, filename)
                with open(model_path, "rb") as f:
                    trained_models[model_name] = pickle.load(f)
                print(f"Modèle chargé : {model_name}")
        return trained_models



    def generate_summary_table(self):
        """Génère un tableau récapitulatif des performances des modèles et le sauvegarde en CSV."""
        df = pd.DataFrame.from_dict(self.results, orient='index')
        save_path = os.path.join(self.save_dir, "summary_table.csv")
        df.to_csv(save_path, index=True, sep=',', encoding='utf-8')
        print(f"Tableau récapitulatif sauvegardé : {save_path}")

        # Sauvegarde en PNG
        save_png_path = os.path.join(self.save_dir, "summary_table.png")
        plt.figure(figsize=(10, 4))
        sns.heatmap(df, annot=True, fmt=".4f", cmap="Blues", linewidths=0.5)
        plt.title("Tableau récapitulatif des performances des modèles")
        plt.savefig(save_png_path, bbox_inches='tight')
        print(f"Tableau récapitulatif sauvegardé en image : {save_png_path}")
        plt.close()




    def plot_individual_metric(self, metric, values):
        """Génère un graphique pour une métrique spécifique."""
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(values.keys()), y=list(values.values()), hue=list(values.keys()), palette="viridis", legend=False)
        plt.xlabel("Modèles")
        plt.ylabel("Score")
        plt.title(f"Comparaison des performances - {metric}")
        plt.xticks(rotation=45)
        save_path = os.path.join(self.metric_dirs[metric], f"{metric}_comparaison.png")
        plt.savefig(save_path)
        print(f"Graphique enregistré : {save_path}")
        plt.close()

    def plot_combined_metrics(self):
        """ Génère un graphique unique combinant les 4 métriques."""
        plt.figure(figsize=(12, 6))
        
        models = list(self.results.keys())
        metrics = ["TP_Rate", "FP_Rate", "F1-score", "AUC"]
        colors = ["blue", "red", "green", "purple"]
        markers = ["o", "s", "D", "*"]
        
        for metric, color, marker in zip(metrics, colors, markers):
            scores = [self.results[model][metric] for model in models]
            plt.plot(models, scores, marker=marker, linestyle="-", linewidth=2, markersize=8, label=metric, color=color)
        
        plt.xlabel("Modèles")
        plt.ylabel("Score")
        plt.title("Comparaison globale des performances des modèles")
        plt.legend()
        
        save_path = os.path.join(self.save_dir, "Comparaison_globale.png")
        plt.savefig(save_path)
        plt.close()

    def plot_metrics(self):
        metrics = ["TP_Rate", "FP_Rate", "F1-score", "AUC"]
        
        for metric in metrics:
            values = {model: self.results[model][metric] for model in self.results}
            self.plot_individual_metric(metric, values)
        
        self.plot_combined_metrics()

    def evaluate_models(self, X_test, y_test):
        print("\nDébut de l'évaluation des modèles...")
        if not self.trained_models:
            print("Aucun modèle chargé ! Vérifiez le dossier Datatest/Tache3/Entrainement/")
            return
        
        print("Modèles disponibles pour l'évaluation :", list(self.trained_models.keys()))

        for name, model in self.trained_models.items():
            print(f"Évaluation du modèle {name}...")
            
            try:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
                fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
                auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
                
                self.results[name] = {
                    "TP_Rate": tp_rate,
                    "FP_Rate": fp_rate,
                    "F1-score": f1,
                    "AUC": auc
                }
                print(f"Résultats enregistrés pour {name} : {self.results[name]}")
            
            except Exception as e:
                print(f"Erreur lors de l'évaluation du modèle {name} : {e}")

        print("Évaluation terminée !")
        self.generate_summary_table()
        self.plot_metrics()

# Chargement des données
print("Début du chargement des données...")
loader = DataLoader("Datatest/Tache2_donnees_final/data_final.csv")
X_test, y_test = loader.X_test, loader.y_test
print("Données chargées avec succès.")

# Chargement des modèles
print("Début du chargement des modèles...")
model_dir = "Datatest/Tache3/Entrainement"
trained_models = ModelEvaluator.load_models(model_dir)

# Vérification et évaluation des modèles
if trained_models:
    print("* Modèles trouvés, lancement de l'évaluation...")
    evaluator = ModelEvaluator(trained_models)
    evaluator.evaluate_models(X_test, y_test)
else:
    print("--- Aucun modèle trouvé, évaluation annulée !")
''












