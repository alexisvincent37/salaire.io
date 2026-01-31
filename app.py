from dash import Dash
import dash_bootstrap_components as dbc
from src.salarypred.layout import *
from src.salarypred.callback import *

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap"],
    suppress_callback_exceptions=True
)

app.layout = layout

server = app.server

if __name__ == "__main__":
    app.run(debug=False, port=8050)