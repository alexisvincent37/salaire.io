import sys
from pathlib import Path
from data.machine_learning import text_cleaner, get_train_test_data, _get_all_metrics, lm_model, Model
import pytest
import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, parent_dir)


def test_text_cleaner_basic():
    raw_text = "Élève Ingénieur H/F - Data Scientist!"
    expected = "eleve ingenieur data scientist"
    assert text_cleaner(raw_text) == expected

def test_text_cleaner_edge_cases():
    assert text_cleaner(None) == ""
    assert text_cleaner(123) == ""
    assert text_cleaner("") == ""

def test_text_cleaner_stopwords():
    text = "Poste en CDI urgent h/f à pourvoir"
    cleaned = text_cleaner(text)
    assert "cdi" not in cleaned
    assert "urgent" not in cleaned
    assert "h/f" not in cleaned

def test_get_all_metrics():
    y_true = np.array([10, 100])
    y_pred = np.array([10, 110])
    
    metrics = _get_all_metrics(y_true, y_pred)
    
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics['mae'] == 5.0
    assert metrics['rmse'] == pytest.approx(7.071, 0.001)


@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        'titre': ['Data Scientist', 'Data Analyst', 'Dev', 'Lead', 'CTO'],
        'experience': [2, 5, 1, 10, 8],
        'statut': ['cadre'] * 5,
        'teletravail': ['oui'] * 5,
        'region': ['Paris'] * 5,
        'secteur': ['IT'] * 5,
        'metier': ['Data'] * 5,
        'contrat': ['CDI'] * 5,
        'salaire': [30000, 40000, 35000, 60000, 80000]
    })

def test_get_train_test_data(dummy_df):
    X_train, X_test, y_train, y_test = get_train_test_data(dummy_df, test_size=0.2)
    
    assert len(X_train) + len(X_test) == 5
    assert len(y_train) + len(y_test) == 5
    
    assert y_train.mean() < 20  
    assert y_train.min() > 0
    
    assert 'salaire' not in X_train.columns

def test_model_pipeline_integration(dummy_df):
    """
    Smoke test : lance un modèle linéaire complet sur des données dummy.
    """
    large_df = pd.concat([dummy_df] * 4, ignore_index=True)
    
    X_train, X_test, y_train, y_test = get_train_test_data(large_df, test_size=0.4)
    

    result_model = lm_model(X_train, y_train, X_test, y_test)
    
    assert isinstance(result_model, Model)
    assert result_model.name == "LinearRegression"
    assert 'rmse' in result_model.metrics['test']