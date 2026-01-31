# 💰 Salaire.io - Prédicteur de Salaire Data Science

### Projet de Prédiction de Salaires - Offres d'Emploi APEC
Ce projet implémente une chaîne de traitement de données complète allant du nettoyage des données à la mise en production d'un modèle de Machine Learning via une interface web interactive.

L'objectif principal est d'estimer la fourchette salariale d'une offre d'emploi en fonction de ses caractéristiques intrinsèques (intitulé du poste, niveau d'expérience, localisation, type de contrat, etc.), en se basant sur des données extraites du portail de l'APEC.

### La Problématique : Comment estimer de manière fiable la fourchette salariale d’une offre d’emploi lorsque celle-ci ne communique pas explicitement la rémunération, en s’appuyant uniquement sur les caractéristiques textuelles et structurelles de l’offre (poste, expérience, localisation, type de contrat), malgré l’hétérogénéité et l’incomplétude des données ?
Pour un étudiant ou un jeune diplômé s'apprêtant à entrer sur le marché du travail, la navigation sur les sites d'annonce révèle une difficulté majeure : une proportion importante d'offres ne mentionne pas de salaire ou utilise des termes vagues ("selon profil").

Cette opacité empêche les candidats de :

Connaître leur valeur réelle sur le marché.

Comparer efficacement plusieurs offres pour prioriser leurs candidatures.

Préparer sereinement la phase de négociation salariale lors des entretiens.

Scénario d'Utilisation : Aide au choix du premier emploi
Dans ce contexte, le projet sert d'outil d'aide à la décision. Un étudiant peut saisir les détails d'une offre qui l'intéresse mais qui n'affiche pas de rémunération. Le modèle lui fournit une estimation basée sur les tendances actuelles du marché. En comparant les prédictions pour différentes offres, l'utilisateur peut identifier celles qui offrent les meilleures perspectives financières par rapport à son profil et sa localisation, facilitant ainsi un choix de carrière éclairé.
---

## 👥 L'Équipe

Projet réalisé par :
* **Jawad GRIB**
* **Abdul BOLOGOUN**
* **Alexis VINCENT**

---

## 🚀 Fonctionnalités Clés

Le projet couvre l'intégralité de la chaîne de valeur de la donnée :

1.  **Acquisition (Scraping) :** Récupération automatisée d'annonces via l'API de l'APEC (gestion des tokens, requêtes JSON).
2.  **Nettoyage (Preprocessing) :** Traitement des valeurs manquantes, nettoyage des intitulés de poste, standardisation des salaires bruts.
3.  **Modélisation (Machine Learning) :** Comparaison de plusieurs modèle de régression et sélection du meilleur compromis performance/overfitting.
4.  **Visualisation (Web App) :** Interface utilisateur interactive (Dash) permettant de simuler un salaire et de visualiser sa position sur le marché.

---

## 🛠 Stack Technique

Le projet repose sur un écosystème Python complet :

* **Web & Dashboard :** `Dash`, `Dash Bootstrap Components`
* **Data Manipulation :** `Pandas`, `NumPy`
* **Machine Learning :** `Scikit-learn`, `XGBoost`, `Joblib`
* **Scraping :** `Selenium`, `Webdriver-manager`
* **Visualisation :** `Matplotlib`, `Seaborn`
* **Qualité & Tests :** `Pytest`, `Ydata-profiling`, `Rich`

---

## 📂 Architecture du Projet

Voici l'organisation du code source :

```text
salaire.io/
├── data/                       # Scripts de traitement de données
│   ├── scrapping.py            # Récupération des données APEC
│   ├── cleaningdata.py         # Nettoyage et transformation
│   ├── Analyse_de_données.py   # Exploration statistique
|   ├── machine_learning.py     # Pipelines Scikit-learn (Preprocessing)
|   ├── modviz.py               # Visualisation des performances et entrainement des modèles
│   ├── mod/                    # Dossier de sauvegarde du modèle (.joblib)
│   └── dataframe/              # Stockage des CSV (clean & raw)
|   
├── tests/                      # Tests unitaires
│
├── src/                        # Code source de l'application Web
│   ├── salarypred/             # Modules de l'application
│   │   ├── layout.py           # Interface visuelle (HTML/Bootstrap)
│   │   └── callback.py         # Logique interactive et prédictions
│
└── app.py                      # Point d'entrée principal (Main)
```

⚙️ Installation et Lancement
Pour tester le projet en local, suivez ces étapes :

1. Cloner le dépôt :
   ```bash
    git clone https://github.com/alexisvincent37/salaire.io.git
    cd salaire.io
   ```

2. Installer les dépendances :
   ```bash
   pip install dash pytest matplotlib pandas joblib seaborn numpy scikit-learn xgboost rich selenium dash_bootstrap_components webdriver-manager ydata_profiling
   ```
3. Lancer l'application :
   ```bash
    python app.py
    ```

L'application sera accessible à l'adresse : `http://127.0.0.1:8050/`


## 📊 Choix du Modèle et Performance

L'évaluation et la comparaison des algorithmes ont été réalisées via le script `modviz.py`, qui utilise la librairie **Rich** pour générer des tableaux de bord de performance directement dans le terminal.

Après benchmark (`compare_mod.py`), nous avons retenu un **Modèle Linéaire (Linear Regression)** appliqué sur le logarithme du salaire.

* **Score ($R^2$) :** `0.53`
* **Pourquoi ce choix ?**
  Nous avons privilégié la **performance généralisable**. Si des modèles complexes (Random Forest, XGBoost) offraient des résultats bruts similaires ou supérieurs, ils présentaient un risque plus élevé d'**overfitting** (sur-apprentissage).
* **Analyse :**
  Le fait que le modèle linéaire performe aussi bien démontre que la structure des données est intrinsèquement liée à des **interactions linéaires** entre les variables (Expérience, Métier, Région) et le salaire. Le LM capture l'essentiel du signal sans le bruit.
---

*Université de Tours - M2 MECEN - 2025*
