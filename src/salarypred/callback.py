import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dash import Input, Output, State, html, no_update, callback
import dash_bootstrap_components as dbc
import data.machine_learning

def get_resource_path(relative_path):
    """Obtient le chemin absolu, compatible dev et PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent.parent.parent
    return base_path / relative_path

current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir.parent.parent 
sys.path.append(str(project_root))
sys.path.append(str(project_root / "data"))

MODEL_PATH = get_resource_path("data/mod/res_lm.joblib")

def load_model() -> Optional[Any]:
    """
    Charge le modèle de Machine Learning depuis le fichier .joblib.

    Returns:
        Optional[Any]: L'objet pipeline Scikit-learn chargé ou None en cas d'échec.
    """
    if MODEL_PATH.exists():
        try:
            container = joblib.load(MODEL_PATH)
            return container.model_object if hasattr(container, 'model_object') else container
        except Exception:
            return None
    return None

loaded_pipeline = load_model()

@callback(
    Output("modal_input", "is_open"),
    [Input("btn_open_modal", "n_clicks"), Input("btn_validate_finish", "n_clicks")],
    [State("modal_input", "is_open")],
)
def toggle_modal(n_open: Optional[int], n_finish: Optional[int], is_open: bool) -> bool:
    """
    Gère l'ouverture et la fermeture de la fenêtre modale de saisie.

    Args:
        n_open (Optional[int]): Nombre de clics sur le bouton d'ouverture.
        n_finish (Optional[int]): Nombre de clics sur le bouton de validation.
        is_open (bool): État actuel de la modale.

    Returns:
        bool: Nouvel état d'ouverture de la modale.
    """
    if n_open or n_finish:
        return not is_open
    return is_open

@callback(
    [
        Output("job_store", "data"),
        Output("job_list_container", "children"),
        Output("input_titre", "value"),
        Output("dd_metier", "value"),
        Output("input_exp", "value"),
        Output("dd_secteur", "value"),
        Output("dd_region", "value"),
        Output("dd_statut", "value"),
        Output("dd_contrat", "value"),
        Output("dd_teletravail", "value"),
    ],
    [Input("btn_add_continue", "n_clicks"), Input("btn_validate_finish", "n_clicks")],
    [
        State("input_titre", "value"),
        State("dd_metier", "value"),
        State("input_exp", "value"),
        State("dd_secteur", "value"),
        State("dd_region", "value"),
        State("dd_statut", "value"),
        State("dd_contrat", "value"),
        State("dd_teletravail", "value"),
        State("job_store", "data")
    ]
)
def update_with_prediction(
    n_add: Optional[int], 
    n_finish: Optional[int], 
    titre: Optional[str], 
    metier: Optional[str], 
    exp: Optional[int], 
    sect: Optional[str], 
    reg: Optional[str], 
    stat: Optional[str], 
    cont: Optional[str], 
    tt: Optional[str], 
    current_data: Optional[List[Dict[str, Any]]]
) -> Tuple[Any, ...]:
    """
    Traite la saisie utilisateur, effectue une prédiction de salaire et met à jour l'interface.

    Args:
        n_add (Optional[int]): Clics bouton 'Ajouter et Continuer'.
        n_finish (Optional[int]): Clics bouton 'Valider et Terminer'.
        titre (Optional[str]): Titre du poste saisi.
        metier (Optional[str]): Métier sélectionné.
        exp (Optional[int]): Années d'expérience.
        sect (Optional[str]): Secteur d'activité.
        reg (Optional[str]): Région.
        stat (Optional[str]): Statut (Cadre, etc.).
        cont (Optional[str]): Type de contrat.
        tt (Optional[str]): Politique de télétravail.
        current_data (Optional[List[Dict[str, Any]]]): Données actuellement stockées dans le dcc.Store.

    Returns:
        Tuple[Any, ...]: Tuple contenant les données mises à jour, le contenu HTML du tableau, 
                         et les valeurs remises à zéro pour le formulaire.
    """
    if not n_add and not n_finish:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    current_data = current_data or []
    
    if titre or metier:
        salaire_predit = 0.0
        
        input_df = pd.DataFrame([{
            'titre': str(titre) if titre else "Employé",
            'metier': str(metier) if metier else "Autre",
            'experience': float(exp) if exp else 0.0,
            'secteur': str(sect) if sect else "Autre",
            'region': str(reg) if reg else "AUTRE REGION",
            'statut': str(stat) if stat else "Non spécifié",
            'contrat': str(cont) if cont else "CDI",
            'teletravail': str(tt) if tt else "Non autorisé"
        }])
        
        if loaded_pipeline:
            try:
                pred_log = loaded_pipeline.predict(input_df)[0]
                salaire_predit = round(np.expm1(pred_log), 1)
            except Exception:
                salaire_predit = 0.0
        
        new_job = {
            "Titre": titre or "Non spécifié",
            "Métier": metier or "-",
            "Expérience": f"{exp} ans" if exp else "0 ans",
            "Localisation": f"{reg}" if reg else "-",
            "Contrat": f"{cont}",
            "Salaire": salaire_predit
        }
        current_data.append(new_job)

    sorted_data = sorted(current_data, key=lambda x: x['Salaire'], reverse=True)

    table_header = [
        html.Thead(html.Tr([
            html.Th("#"),
            html.Th("Poste"),
            html.Th("Détails"),
            html.Th("Contrat"),
            html.Th("Salaire (k€)"),
        ]))
    ]

    rows = []
    for idx, job in enumerate(sorted_data):
        rank = idx + 1
        row_class = "top-1" if rank == 1 else ""
        
        val_display = f"{job['Salaire']} k€" if job['Salaire'] > 0 else "N/A"
        
        rows.append(html.Tr([
            html.Td(html.Div(f"{rank}", className="rank-badge"), className=row_class),
            html.Td(html.Div([
                html.Div(job['Titre'], style={"font-weight": "bold", "color": "white"}),
                html.Div(job['Métier'], style={"font-size": "0.8rem", "color": "#8b9bb4"})
            ])),
            html.Td(f"{job['Expérience']} • {job['Localisation']}"),
            html.Td(job['Contrat']),
            html.Td(html.Span([html.I(className="fas fa-robot me-2"), val_display], className="salary-badge"))
        ]))

    table_content = html.Div([
        html.H2("Estimations du Marché", style={"text-align": "center", "margin-bottom": "20px"}),
        dbc.Table(table_header + [html.Tbody(rows)], hover=True, responsive=True, className="table-dark-custom")
    ], className="ranking-container")

    return sorted_data, table_content, "", None, None, None, None, None, None, None