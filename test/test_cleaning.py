import sys
import os
import pandas as pd
import numpy as np
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from data.cleaningdata import clean_data 

@pytest.fixture
def raw_df():
    return pd.DataFrame({
        'salaire_min': [30000, 45000, np.nan, 20000],
        'salaire_max': [40000, 55000, 60000, np.nan],
        'entreprise': ['Super Boite', 'Boite B', 'Non spécifié (Confidentiel)', 'Boite D'],
        'titre': ['Data Scientist', 'Infirmier', 'Boulanger', 'Commercial'],
        'lieu': ['Paris - 75001', 'Lyon 69', 'Marseille', 'Toulouse'], 
        'metier': ['Développeur Python', 'Infirmier DE', 'Vendeur', 'Ingénieur d\'affaires'],
        'secteur': ['Informatique', 'Santé', 'Commerce', 'Vente'],
        'experience': ['2 ans', '5', 'débutant', '10 ans'],
        'contrat': ['CDI', 'CDD (6 mois)', 'Mission intérim', 'Stage'],
        'teletravail': ['oui', 'impossible', np.nan, 'partiel'],
        'statut': ['cadre', 'tam', 'employé', 'cadre'],
        'salaire_brut_texte': ['a', 'b', 'c', 'd'],
        'etudes': ['bac+5', 'bac+3', 'bac', 'bac+5'],
        'url_source': ['http...', 'http...', 'http...', 'http...']
    })

def test_clean_data_structure(raw_df):
    df_res = clean_data(raw_df)
    
    expected_cols = [
        'salaire', 'titre', 'metier', 'experience', 
        'secteur', 'region', 'statut', 'contrat', 'teletravail'
    ]
    
    assert list(df_res.columns) == expected_cols
    assert 'url_source' not in df_res.columns

def test_salary_calculation(raw_df):
    df_res = clean_data(raw_df)
    
    row1 = df_res[df_res['titre'] == 'Data Scientist'].iloc[0]
    assert row1['salaire'] == 35000.0
    
    assert len(df_res) < 4

def test_text_cleaning_mappings(raw_df):
    df_res = clean_data(raw_df)
    
    paris_row = df_res[df_res['titre'] == 'Data Scientist'].iloc[0]
    assert paris_row['region'] == 'ÎLE-DE-FRANCE'
    
    assert paris_row['metier'] == 'TECH / IT'
    
    infirmier_row = df_res[df_res['titre'] == 'Infirmier'].iloc[0]
    assert infirmier_row['contrat'] == 'CDD'

def test_experience_extraction(raw_df):
    df_res = clean_data(raw_df)
    
    row = df_res[df_res['titre'] == 'Data Scientist'].iloc[0]
    assert row['experience'] == 2
    assert isinstance(row['experience'], (int, np.integer))

def test_filtering_company(raw_df):
    df_res = clean_data(raw_df)
    
    assert 'Boulanger' not in df_res['titre'].values
