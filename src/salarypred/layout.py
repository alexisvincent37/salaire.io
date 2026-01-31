from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from pathlib import Path
import sys

def get_resource_path(relative_path):
    """Obtient le chemin absolu, compatible dev et PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        # On est dans le .exe
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).resolve().parent.parent.parent / relative_path

BASE_DIR = get_resource_path("data/dataframe/data_clean.csv")
try:
    df = pd.read_csv(BASE_DIR)
    unique_metier = sorted([x for x in df['metier'].dropna().unique() if isinstance(x, str)])
    unique_secteur = sorted([x for x in df['secteur'].dropna().unique() if isinstance(x, str)])
    unique_region = sorted([x for x in df['region'].dropna().unique() if isinstance(x, str)])
    unique_statut = sorted([x for x in df['statut'].dropna().unique() if isinstance(x, str)])
    unique_contrat = sorted([x for x in df['contrat'].dropna().unique() if isinstance(x, str)])
    unique_teletravail = sorted([x for x in df['teletravail'].dropna().unique() if isinstance(x, str)])
except Exception:
    unique_metier = unique_secteur = unique_region = unique_statut = unique_contrat = unique_teletravail = []

def create_dropdown(id, options, placeholder):
    return dcc.Dropdown(
        id=id,
        options=[{'label': i, 'value': i} for i in options],
        placeholder=placeholder,
        className="dash-dropdown"
    )

modal_form = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Simulateur de Salaire"), close_button=True),
        dbc.ModalBody(
            [
                dbc.Row([
                    dbc.Col([
                        html.Label("Intitulé du poste", className="modal-label"),
                        dbc.Input(id="input_titre", type="text", placeholder="Ex: Data Scientist", className="form-control", style={"background-color": "#23273a", "border": "1px solid #2d3248", "color": "white", "border-radius": "10px"}),
                    ], width=8),
                    dbc.Col([
                        html.Label("Expérience (ans)", className="modal-label"),
                        dbc.Input(id="input_exp", type="number", placeholder="0", min=0, step=1, style={"background-color": "#23273a", "border": "1px solid #2d3248", "color": "white", "border-radius": "10px"}),
                    ], width=4),
                ], className="mb-2"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Métier", className="modal-label"),
                        create_dropdown("dd_metier", unique_metier, "Famille de métier"),
                    ], width=6),
                    dbc.Col([
                        html.Label("Secteur", className="modal-label"),
                        create_dropdown("dd_secteur", unique_secteur, "Secteur d'activité"),
                    ], width=6),
                ], className="mb-2"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Région", className="modal-label"),
                        create_dropdown("dd_region", unique_region, "Localisation"),
                    ], width=6),
                    dbc.Col([
                        html.Label("Télétravail", className="modal-label"),
                        create_dropdown("dd_teletravail", unique_teletravail, "Politique TT"),
                    ], width=6),
                ], className="mb-2"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Statut", className="modal-label"),
                        create_dropdown("dd_statut", unique_statut, "Cadre / Non Cadre"),
                    ], width=6),
                    dbc.Col([
                        html.Label("Contrat", className="modal-label"),
                        create_dropdown("dd_contrat", unique_contrat, "Type de contrat"),
                    ], width=6),
                ], className="mb-2"),
            ]
        ),
        dbc.ModalFooter(
            [
                dbc.Button("Prédire & Continuer", id="btn_add_continue", color="light", outline=True, className="me-2", style={"font-weight": "600"}),
                dbc.Button("Valider & Voir Classement", id="btn_validate_finish", className="btn-modern", style={"padding": "10px 20px"})
            ]
        ),
    ],
    id="modal_input",
    is_open=False,
    centered=True,
    size="lg",
    backdrop="static"
)

layout = html.Div(
    [
        dcc.Store(id='job_store', data=[], storage_type='memory'),
        html.Div(
            [
                html.H1("Salaire.io", className="main-title"),
                html.P("Estimation de salaire par regression.", className="subtitle"),
                dbc.Button("Lancer une estimation", id="btn_open_modal", className="btn-modern", n_clicks=0),
            ],
            className="hero-container",
        ),
        modal_form,
        html.Div(id="job_list_container")
    ]
)