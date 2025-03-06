import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

class Tache4Processor:
    def __init__(self, data_path, test_size=0.2, random_state=42):
        """
        Classe qui effectue toute la Tâche 4 : création du sous-ensemble, entraînement et évaluation.
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.save_dir = "Datatest/Tache4"
        self.train_dir = os.path.join(self.save_dir, "Entrainement")
        self.result_dir = os.path.join(self.save_dir, "Results")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Chargement des données
        self.data = self.load_data()
        self.imbalanced_data = self.create_imbalanced_dataset()
        
        # Séparation des features et de la cible
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()
        
        # Dictionnaire des modèles
        self.models = {
            "DecisionTree": DecisionTreeClassifier(random_state=self.random_state),
            "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=self.random_state),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=self.random_state),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=self.random_state),
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=self.random_state),
            "NaiveBayes": GaussianNB()
        }

    def load_data(self):
        """Charge les données finales."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Le fichier {self.data_path} est introuvable.")
        return pd.read_csv(self.data_path, sep=',', encoding='utf-8')
    
    def create_imbalanced_dataset(self):
        """Crée un sous-ensemble déséquilibré avec 5% de pollueurs."""
        polluters = self.data[self.data['Classe'] == 1]
        legitimate = self.data[self.data['Classe'] == 0]
        
        num_legitimate = min(10000, len(legitimate))
        num_polluters = int(num_legitimate * 0.05)
        
        legitimate_sample = legitimate.sample(n=num_legitimate, random_state=self.random_state, replace=False)
        polluter_sample = polluters.sample(n=num_polluters, random_state=self.random_state, replace=False)
        
        imbalanced_data = pd.concat([legitimate_sample, polluter_sample]).sample(frac=1, random_state=self.random_state)
        save_path = os.path.join(self.save_dir, "imbalanced_data.csv")
        imbalanced_data.to_csv(save_path, index=False, sep=',', encoding='utf-8')
        return imbalanced_data
    
    def split_train_test(self):
        """Sépare les données déséquilibrées en train/test."""
        X = self.imbalanced_data.drop(columns=['Classe'])
        y = self.imbalanced_data['Classe']
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

    def train_models(self):
        """Entraîne et sauvegarde les modèles."""
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            model_path = os.path.join(self.train_dir, f"{name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    
    def evaluate_models(self):
        """Évalue les modèles et génère les graphiques."""
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else None
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            f1 = classification_report(self.y_test, y_pred, output_dict=True)['1']['f1-score']
            auc = roc_auc_score(self.y_test, y_prob) if y_prob is not None else np.nan
            results[name] = {"TP_Rate": tp_rate, "FP_Rate": fp_rate, "F1-score": f1, "AUC": auc}

        # Sauvegarde CSV
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_path = os.path.join(self.result_dir, "model_performance.csv")
        results_df.to_csv(results_path, index=True, sep=',', encoding='utf-8')

        # Génération des graphiques individuels
        for metric in ["TP_Rate", "FP_Rate", "F1-score", "AUC"]:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=results_df.index, y=results_df[metric], hue=results_df.index, palette="viridis", legend=False)
            plt.xticks(rotation=45)
            plt.title(f"Comparaison des performances - {metric}")
            save_path = os.path.join(self.result_dir, f"{metric}_comparaison.png")
            plt.savefig(save_path)
            plt.close()

        # Graphique global comparatif
        plt.figure(figsize=(12, 6))
        models = list(results_df.index)
        colors = ["blue", "red", "green", "purple"]
        markers = ["o", "s", "D", "*"]
        
        for metric, color, marker in zip(["TP_Rate", "FP_Rate", "F1-score", "AUC"], colors, markers):
            scores = results_df[metric].tolist()
            plt.plot(models, scores, marker=marker, linestyle="-", linewidth=2, markersize=8, label=metric, color=color)
        
        plt.xlabel("Modèles")
        plt.ylabel("Score")
        plt.title("Comparaison globale des performances des modèles")
        plt.legend()
        
        save_path = os.path.join(self.result_dir, "comparaison_globale.png")
        plt.savefig(save_path)
        plt.close()

        # Tableau en PNG
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=results_df.round(4).values, 
                         colLabels=results_df.columns,
                         rowLabels=results_df.index,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        table_path = os.path.join(self.result_dir, "model_performance.png")
        plt.savefig(table_path, bbox_inches='tight')
        plt.close()

# Exécution de la Tâche 4
if __name__ == "__main__":
    processor = Tache4Processor("Datatest/Tache2_donnees_final/data_final.csv")
    processor.train_models()
    processor.evaluate_models()
    print("************* Tâche 4 terminée avec succès !***********")
