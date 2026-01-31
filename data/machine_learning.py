import pandas as pd
import numpy as np
import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

def text_cleaner(text: Any) -> str:
    """
    Nettoie et normalise une chaîne de caractères (NLP basique).
    
    Supprime les accents, met en minuscules, retire certains mots-clés RH 
    (h/f, cdi, etc.) et ne garde que les lettres.

    Args:
        text (Any): Texte brut à nettoyer.

    Returns:
        str: Texte nettoyé.
    """
    if not isinstance(text, str): return ""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'\b(h/f|hf|cdi|cdd|urgent|f/h|f h)\b', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join(text.split())

def creer_preprocesseur() -> ColumnTransformer:
    """
    Configure le pipeline de prétraitement des données.
    
    - Numérique : StandardScaler sur 'experience'.
    - Catégoriel : OneHotEncoder sur statut, region, etc.
    - Texte : TfidfVectorizer sur 'titre'.

    Returns:
        ColumnTransformer: Le processeur scikit-learn configuré.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['experience']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['statut', 'teletravail', 'region', 'secteur', 'metier', 'contrat']),
            ('text', TfidfVectorizer(max_features=150, preprocessor=text_cleaner), 'titre')
        ],
        remainder='drop'
    )

def get_train_test_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sépare les données en Train/Test et applique une log-transformation sur la cible (salaire).

    Args:
        df (pd.DataFrame): DataFrame source.
        test_size (float): Proportion du jeu de test (défaut 0.1).

    Returns:
        Tuple: x_train, x_test, y_train, y_test.
    """
    X = df.drop(columns=['salaire'])
    y = np.log1p(df['salaire'])
    return train_test_split(X, y, test_size=test_size)

class Model:
    """
    Structure de données pour stocker un modèle et ses performances.
    """
    def __init__(self, name: str, model_object: BaseEstimator, cv_scores: List[float], 
                 train_preds: np.ndarray, metrics: Dict[str, Dict[str, float]], 
                 parameters: Optional[Dict[str, Any]] = None, grid_history: Optional[pd.DataFrame] = None):
        """
        Args:
            name (str): Nom du modèle (ex: 'RandomForest_1').
            model_object (BaseEstimator): L'estimateur scikit-learn entraîné.
            cv_scores (List[float]): Liste des scores de validation croisée (MSE).
            train_preds (np.ndarray): Prédictions sur le train set (échelle réelle).
            metrics (Dict): Dictionnaire des métriques {'train': {...}, 'test': {...}}.
            parameters (Dict, optional): Meilleurs hyperparamètres trouvés.
            grid_history (pd.DataFrame, optional): Historique du GridSearch.
        """
        self.name = name
        self.model_object = model_object
        self.cv_scores = cv_scores
        self.train_preds = train_preds
        self.test_preds = None
        self.metrics = metrics
        self.parameters = parameters
        self.grid_history = grid_history if grid_history is not None else pd.DataFrame()

cv_structure = KFold(n_splits=10, shuffle=True)

def _get_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcule RMSE, MAE, MAPE et R2."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def train_model_cv(name: str, estimator: BaseEstimator, params: Dict[str, Any], 
                   x_train: pd.DataFrame, y_train: pd.Series) -> Model:
    """
    Phase 1 : Entraînement et Validation Croisée (sans toucher au Test Set).
    
    Effectue un GridSearchCV si des paramètres sont fournis, sinon entraîne simplement.
    Récupère les scores CV et calcule les métriques sur le Train set.

    Args:
        name (str): Nom à donner au modèle.
        estimator (BaseEstimator): L'algo scikit-learn (ex: RandomForestRegressor()).
        params (Dict): Grille d'hyperparamètres (ex: {'reg__n_estimators': [100, 200]}).
        x_train, y_train: Données d'entraînement.

    Returns:
        Model: Objet contenant le modèle entraîné et les stats CV/Train.
    """
    prep = creer_preprocesseur()
    pipe = Pipeline([('prep', prep), ('reg', estimator)])
    
    # GridSearchCV pour l'optimisation et la CV
    grid = GridSearchCV(pipe, params, cv=cv_structure, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(x_train, y_train)
    
    best_est = grid.best_estimator_
    
    # Extraction de l'historique GridSearch
    full_history = pd.DataFrame(grid.cv_results_)
    best_idx = grid.best_index_
    n_splits = grid.cv.get_n_splits(x_train, y_train)
    
    # Récupération des scores CV (négatifs par défaut dans sklearn, on inverse)
    cv_scores_log = [-full_history.loc[best_idx, f'split{i}_test_score'] for i in range(n_splits)]
    
    # Nettoyage historique pour affichage
    cols = [c for c in full_history.columns if 'param_' in c] + ['mean_test_score', 'std_test_score']
    clean_history = full_history[cols].sort_values('mean_test_score', ascending=False)
    clean_history['mean_test_score'] = -clean_history['mean_test_score']

    # Prédictions Train (vérif overfitting)
    train_preds_log = best_est.predict(x_train)
    train_preds_real = np.expm1(train_preds_log)
    y_train_real = np.expm1(y_train)

    return Model(
        name=name,
        model_object=best_est,
        cv_scores=cv_scores_log,
        train_preds=train_preds_real,
        metrics={'train': _get_all_metrics(y_train_real, train_preds_real)},
        parameters=grid.best_params_,
        grid_history=clean_history
    )

def evaluate_model_test(model: Model, x_test: pd.DataFrame, y_test: pd.Series) -> Model:
    """
    Phase 2 : Évaluation finale sur le Test Set.
    
    Ne doit être appelé qu'une fois le modèle sélectionné comme champion.

    Args:
        model (Model): L'objet modèle pré-entraîné.
        x_test, y_test: Données de test.

    Returns:
        Model: Le même objet mis à jour avec les métriques de test.
    """
    test_preds_log = model.model_object.predict(x_test)
    test_preds_real = np.expm1(test_preds_log)
    y_test_real = np.expm1(y_test)
    
    model.test_preds = test_preds_real
    model.metrics['test'] = _get_all_metrics(y_test_real, test_preds_real)
    return model

def get_model_config() -> Dict[str, Tuple[BaseEstimator, Dict[str, str]]]:
    """
    Retourne la configuration des modèles disponibles et le schéma de leurs hyperparamètres.
    
    Returns:
        Dict: Clé = Nom algo, Valeur = (Instance Algo, {ParamNom: TypeInput})
    """
    return {
        'LinearRegression': (LinearRegression(), {}),
        'Ridge': (Ridge(), {'reg__alpha': 'float_list'}),
        'Lasso': (Lasso(), {'reg__alpha': 'float_list'}),
        'ElasticNet': (ElasticNet(), {'reg__alpha': 'float_list', 'reg__l1_ratio': 'float_list'}),
        'DecisionTree': (DecisionTreeRegressor(), {'reg__max_depth': 'int_list', 'reg__min_samples_split': 'int_list'}),
        'RandomForest': (RandomForestRegressor(n_jobs=-1), {'reg__n_estimators': 'int_list', 'reg__max_depth': 'int_list', 'reg__min_samples_split': 'int_list', 'reg__max_features': 'str_list'}),
        'XGBoost': (XGBRegressor(n_jobs=-1), {'reg__learning_rate': 'float_list', 'reg__n_estimators': 'int_list', 'reg__max_depth': 'int_list', 'reg__subsample': 'float_list', 'reg__alpha': 'float_list'}),
        'SVR': (SVR(), {'reg__C': 'float_list', 'reg__kernel': 'str_list'})
    }