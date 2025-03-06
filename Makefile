# Définit l'interpréteur Python
PYTHON = python

# Définition des scripts à exécuter
FEATURE_SCRIPT = feature.py
PREPROCESSING_SCRIPT = preprocessing.py
DATA_PREPARATION_SCRIPT = data_preparation.py
DATA_FINAL_SCRIPT = data_final.py
DATA_LOADER_SCRIPT = data_loader.py
MODEL_TRAINER_SCRIPT = model_trainer.py
MODEL_EVALUATOR_SCRIPT = model_evaluator.py
TACHE4_PROCESSOR_SCRIPT = tache4_processor.py
COMPARISON_SCRIPT = comparaisonTache.py 

# Dossiers utilisés
RESULTS_DIR_TACHE3 = Datatest/Tache3/Results
MODELS_DIR_TACHE3 = Datatest/Tache3/Entrainement
RESULTS_DIR_TACHE4 = Datatest/Tache4/Results
MODELS_DIR_TACHE4 = Datatest/Tache4/Entrainement
COMPARISON_DIR = Datatest/Comparaison_Tache3_Tache4

# Cible principale : exécuter toutes les étapes dans l'ordre
all: install_deps feature preprocessing data_preparation data_final data_loader model_trainer model_evaluator tache4 comparison

# Installation des dépendances (avec vérification de requirements.txt)
install_deps:
	@echo "* Vérification de requirements.txt..."
	@if [ ! -f requirements.txt ]; then \
		echo "---------  requirements.txt introuvable, génération automatique... -------"; \
		$(PYTHON) -m pip freeze > requirements.txt; \
	fi
	@echo "* Installation des dépendances..."
	$(PYTHON) -m pip install -r requirements.txt || true
	@echo "* Vérification et installation des paquets manquants..."
	$(PYTHON) -m pip install numpy pandas scikit-learn matplotlib scikit-plot seaborn --no-cache-dir


# Étape 1 : Extraction des caractéristiques (feature.py)
feature:
	@echo "-- Étape 1 : Extraction des caractéristiques..."
	$(PYTHON) $(FEATURE_SCRIPT)

# Étape 2 : Prétraitement des données (preprocessing.py)
preprocessing: feature
	@echo "-- Étape 2 : Prétraitement des données..."
	$(PYTHON) $(PREPROCESSING_SCRIPT)

# Étape 3 : Nettoyage et normalisation des données (data_preparation.py)
data_preparation: preprocessing
	@echo "-- Étape 3 : Nettoyage et normalisation des données..."
	$(PYTHON) $(DATA_PREPARATION_SCRIPT)

# Étape 4 : Fusion des données finales (data_final.py)
data_final: data_preparation
	@echo "-- Étape 4 : Fusion des données finales..."
	$(PYTHON) $(DATA_FINAL_SCRIPT)

# Étape 5 : Chargement des données (data_loader.py)
data_loader: data_final
	@echo "-- Étape 5 : Chargement des données..."
	$(PYTHON) $(DATA_LOADER_SCRIPT)

# Étape 6 : Entraînement des modèles (model_trainer.py)
model_trainer: data_loader
	@echo "-- Étape 6 : Entraînement des modèles..."
	$(PYTHON) $(MODEL_TRAINER_SCRIPT)

# Étape 7 : Évaluation des modèles (model_evaluator.py)
model_evaluator: model_trainer
	@echo "-- Étape 7 : Évaluation des modèles..."
	$(PYTHON) $(MODEL_EVALUATOR_SCRIPT)

# Étape 8 : Tâche 4 - Analyse comparative sur classes déséquilibrées
tache4: model_evaluator
	@echo "-- Étape 8 : Exécution de la Tâche 4 (Données déséquilibrées)..."
	$(PYTHON) $(TACHE4_PROCESSOR_SCRIPT)
	@echo ""
	@echo "+++++++++++++ TÂCHE 4 TERMINÉE AVEC SUCCÈS ! ++++++++++++++++++"
	@echo ""

# Étape 9 : Comparaison des performances entre la Tâche 3 et la Tâche 4
comparison: tache4
	@echo "-- Étape 9 : Génération des graphiques comparatifs entre la Tâche 3 et la Tâche 4..."
	$(PYTHON) $(COMPARISON_SCRIPT)
	@echo ""
	@echo "+++++++++++++ COMPARAISON ENTRE TÂCHE 3 ET TÂCHE 4 TERMINÉE ! ++++++++++++++++++"
	@echo ""

	# Affichage d'un message final pour macOS et Linux
	@echo "*******************************************************************************************************************************"
	@echo "***************************************** IMPORTANT :**************************************************************************"
	@echo "* Pour OUVRIR les graphes de COMPARAISON GLOBALE 										*"
	@echo "* utilisez la commande suivante :                									*"
	@echo "*                                                									*"
	@echo "# * <-----------  **open Datatest/Comparaison_Tache3_Tache4/comparaison_globale.png** --------->					*"
	@echo "*                                                									*"
	@echo "* Si vous êtes sous Linux, utilisez :            									*"
	@echo "*   xdg-open Datatest/Comparaison_Tache3_Tache4/comparaison_globale.png  								*"
	@echo "*******************************************************************************************************************************"
	@echo ""

	# Affichage d'une boîte de dialogue sur macOS
	@osascript -e 'display dialog "Comparaison terminée !\n\n📊 Pour voir le graphe de comparaison globale, ouvrez :\nDatatest/Comparaison_Tache3_Tache4/comparaison_globale.png\n\nExécutez cette commande :\nopen Datatest/Comparaison_Tache3_Tache4/comparaison_globale.png" with title "Analyse Comparaison Globale" buttons {"OK"} default button "OK"' || true

# Nettoyage des fichiers générés
clean:
	@echo "-Nettoyage- Suppression des fichiers intermédiaires et résultats..."
	rm -rf Datatest/Tache2/Partie1/*.csv Datatest/Tache2/Partie2/*.csv Datatest/Tache2_donnees_final/*.csv
	rm -rf $(RESULTS_DIR_TACHE3) $(MODELS_DIR_TACHE3)
	rm -rf $(RESULTS_DIR_TACHE4) $(MODELS_DIR_TACHE4)
	rm -rf Datatest/
	rm -f requirements.txt
	clear

.PHONY: all install_deps feature preprocessing data_preparation data_final data_loader model_trainer model_evaluator tache4 comparison clean
