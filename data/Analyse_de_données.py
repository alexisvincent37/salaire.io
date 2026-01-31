import pandas as pd
import webbrowser
import tempfile
from ydata_profiling import ProfileReport
from pathlib import Path

FILE_PATH = Path(__file__).resolve().parent / "dataframe" / "data_clean.csv"

df = pd.read_csv(FILE_PATH)

cols_qualitatives = ['metier', 'secteur', 'region', 'statut', 'contrat', 'teletravail']
df[cols_qualitatives] = df[cols_qualitatives].astype('category')

profile = ProfileReport(df, title="Résumé Statistique")

with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
    profile.to_file(tmp.name)
    webbrowser.open(f'file://{tmp.name}')