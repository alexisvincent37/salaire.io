import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Union

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "dataframe" / "income.csv"
SAVE_PATH = BASE_DIR / "dataframe" / "data_clean.csv"

def map_metier(m: Union[str, float]) -> str:
    """
    Associe un intitulé de poste à une catégorie de métier standardisée.

    Args:
        m (Union[str, float]): L'intitulé du poste brut (chaîne de caractères ou NaN).

    Returns:
        str: La catégorie de métier normalisée (ex: 'TECH / IT', 'RH').
    """
    m_str = str(m).lower()
    if any(w in m_str for w in ['informatique', 'développeur', 'système', 'réseaux', 'data', 'analyste', 'test', 'digital', 'bi', 'progiciel']):
        return 'TECH / IT'
    if any(w in m_str for w in ['santé', 'infirmier', 'médecin', 'ehpad', 'pharmacien', 'cadre de santé']):
        return 'SANTÉ / MÉDICAL'
    if any(w in m_str for w in ['rh', 'ressources humaines', 'recrutement', 'formation']):
        return 'RH'
    if any(w in m_str for w in ['juriste', 'juridique', 'avocat', 'fiscaliste', 'notaire']):
        return 'JURIDIQUE'
    if any(w in m_str for w in ['comptable', 'finance', 'audit', 'contrôle de gestion', 'analyste financier', 'trésorerie', 'recouvrement', 'paie']):
        return 'FINANCE / GESTION'
    if any(w in m_str for w in ['commercial', 'technico-commercial', 'ventes', 'comptes', 'business', 'key account']):
        return 'COMMERCIAL'
    if any(w in m_str for w in ['logistique', 'supply chain', 'approvisionneur', 'flux', 'entrepôt', 'transport', 'achats', 'acheteur']):
        return 'LOGISTIQUE / ACHATS'
    if any(w in m_str for w in ['marketing', 'communication', 'chef de produit', 'évènementiel']):
        return 'MARKETING / COM'
    if any(w in m_str for w in ['ingénieur', 'process', 'qualité', 'hse', 'cvc', 'électronique', 'électrique', 'calcul', 'structure', 'mécanique', 'r&d', 'travaux', 'chantier']):
        return 'INGÉNIERIE / INDUSTRIE'
    if any(w in m_str for w in ['directeur', 'responsable', 'chef de service', 'management']):
        return 'MANAGEMENT / DIRECTION'
    return 'AUTRES / SERVICES'

def map_secteur(s: Union[str, float]) -> str:
    """
    Associe un secteur d'activité brut à une catégorie standardisée.

    Args:
        s (Union[str, float]): Le nom du secteur brut (chaîne de caractères ou NaN).

    Returns:
        str: Le groupe de secteur normalisé (ex: 'TECH / IT', 'FINANCE / DROIT').
    """
    s_str = str(s).upper()
    if any(w in s_str for w in ['INFORMATIQUE', 'LOGICIEL', 'DONNÉES', 'PROGRAMMATION']):
        return 'TECH / IT'
    if any(w in s_str for w in ['CONSEIL', 'GESTION', 'SIÈGES SOCIAUX', 'HOLDING']):
        return 'CONSEIL / GESTION'
    if any(w in s_str for w in ['PLACEMENT', 'TRAVAIL TEMPORAIRE', 'RESSOURCES HUMAINES']):
        return 'RH / RECRUTEMENT'
    if any(w in s_str for w in ['INGÉNIERIE', 'TECHNIQUES', 'FABRICATION', 'CONSTRUCTION', 'INDUSTRIE']):
        return 'INDUSTRIE / INGÉNIERIE'
    if any(w in s_str for w in ['HOSPITAL', 'SANTÉ', 'PHARMACEUTIQUE', 'ÂGÉES', 'HANDICAP', 'MÉDICAL']):
        return 'SANTÉ / SOCIAL'
    if any(w in s_str for w in ['ADMINISTRATION PUBLIQUE', 'SÉCURITÉ SOCIALE', 'ENSEIGNEMENT', 'FORMATION']):
        return 'PUBLIC / ÉDUCATION'
    if any(w in s_str for w in ['COMMERCE', 'TRANSPORT', 'ENTREPOSAGE', 'ACHAT', 'LOGISTIQUE', 'AFFRÈTEMENT']):
        return 'COMMERCE / LOGISTIQUE'
    if any(w in s_str for w in ['COMPTABLE', 'ASSURANCE', 'FINANCIER', 'JURIDIQUE', 'BANQUE']):
        return 'FINANCE / DROIT'
    return 'AUTRES'

def map_region(v: Union[str, float]) -> str:
    """
    Associe une ville ou un lieu géographique à une région administrative française.

    Args:
        v (Union[str, float]): Le lieu ou la ville (chaîne de caractères ou NaN).

    Returns:
        str: Le nom de la région normalisé (ex: 'ÎLE-DE-FRANCE').
    """
    v_str = str(v).lower()
    if any(w in v_str for w in ['paris', 'pontoise', 'bezons', 'évry', 'nanterre', 'créteil', 'versailles', 'boulogne', 'saint-denis']): return 'ÎLE-DE-FRANCE'
    if any(w in v_str for w in ['lyon', 'grenoble', 'saint-étienne', 'clermont', 'annecy', 'valence']): return 'AUVERGNE-RHÔNE-ALPES'
    if any(w in v_str for w in ['marseille', 'aix', 'nice', 'toulon', 'avignon', 'cannes']): return 'PACA'
    if any(w in v_str for w in ['toulouse', 'montpellier', 'nîmes', 'perpignan']): return 'OCCITANIE'
    if any(w in v_str for w in ['lille', 'amiens', 'roubaix', 'tourcoing', 'dunkerque']): return 'HAUTS-DE-FRANCE'
    if any(w in v_str for w in ['bordeaux', 'limoges', 'poitiers', 'pau', 'la rochelle']): return 'NOUVELLE-AQUITAINE'
    if any(w in v_str for w in ['nantes', 'angers', 'le mans', 'saint-nazaire']): return 'PAYS DE LA LOIRE'
    if any(w in v_str for w in ['rennes', 'brest', 'quimper', 'lorient']): return 'BRETAGNE'
    if any(w in v_str for w in ['strasbourg', 'reims', 'metz', 'nancy', 'mulhouse']): return 'GRAND EST'
    if any(w in v_str for w in ['rouen', 'caen', 'le havre']): return 'NORMANDIE'
    if any(w in v_str for w in ['dijon', 'besançon']): return 'BOURGOGNE-FRANCHE-COMTÉ'
    return 'AUTRE REGION'

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Effectue le nettoyage complet du DataFrame des salaires.

    Args:
        df (pd.DataFrame): Le DataFrame brut contenant les données brutes.

    Returns:
        pd.DataFrame: Le DataFrame nettoyé avec les colonnes renommées, 
                      les valeurs manquantes traitées et les catégories standardisées.
    """
    df = df.dropna(subset=['salaire_max'])
    df = df.dropna(subset=['salaire_min'])
    df['salaire'] = df[['salaire_min', 'salaire_max']].mean(axis=1)

    df = df.drop(columns=["salaire_brut_texte", 'entreprise', "etudes", "url_source", "salaire_min", "salaire_max"])
    
    df['teletravail'] = df['teletravail'].fillna('Non autorisé')
    df['teletravail'] = df['teletravail'].replace('impossible', 'Non autorisé')
    df = df.sort_values(['teletravail'])
    
    df['contrat'] = df['contrat'].str.extract(r'(CDI|CDD|intérim|stage)', expand=False).fillna('Autre')
    
    df['lieu'] = df['lieu'].str.replace(r' - \d+$', '', regex=True)
    df['lieu'] = df['lieu'].str.replace(r'\d+', '', regex=True)
    df = df.dropna(subset=['lieu'])

    df = df.dropna(subset=['metier'])

    df['experience'] = df['experience'].str.extract(r'(\d+)').fillna(0).astype(int)
    
    df['metier_groupe'] = df['metier'].apply(map_metier)
    df = df.drop(columns=['metier'])

    df['secteur_groupe'] = df['secteur'].apply(map_secteur)
    df = df.drop(columns=['secteur'])

    df['region'] = df['lieu'].apply(map_region)
    df = df.drop(columns=['lieu'])

    df = df.rename(columns={"metier_groupe": "metier", "secteur_groupe": "secteur"})
    df = df.drop_duplicates()

    order = [
        'salaire', 
        'titre',
        'metier', 
        'experience', 
        'secteur', 
        'region', 
        'statut', 
        'contrat', 
        'teletravail'
    ]

    return df[order]

if __name__ == "__main__":
    if FILE_PATH.exists():
        df_raw = pd.read_csv(FILE_PATH)
        df_cleaned = clean_data(df_raw.copy())
        df_cleaned.to_csv(SAVE_PATH, index=False)