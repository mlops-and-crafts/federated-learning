import re
from pathlib import Path
from typing import Union, List, Dict
import logging

import numpy as np
import pandas as pd

from dash import Dash, html, dcc, no_update
from dash.dependencies import Input, Output
import plotly.express as px

import cfg
from metrics import MetricsJSONstore

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("federated-dashboard")


app = Dash(__name__)
app.title = "Federated Learning Dashboard"
app.layout = html.Div([
    html.Div([
        dcc.Graph(id='federated-rmse-graph'),
    ]),
    dcc.Interval(
        id='interval-ticker',
        interval=1000, # in milliseconds
        n_intervals=0
    )
])

@app.callback(
    Output('federated-rmse-graph', 'figure'), 
    Input('interval-ticker', 'n_intervals')
)
def update_data(n_intervals):
    try:
        metrics = MetricsJSONstore(cfg.METRICS_FILE).load()["server"]
        metrics_df = pd.DataFrame(metrics)
        figure = px.line(
            metrics_df, 
            x=metrics_df.server_round, 
            y="rmse", 
            title='RMSE against test data for federated model'
        )
        return figure
    except:
        logger.exception(f"Failed to load and display server metrics from {cfg.METRICS_FILE}")
        return no_update

if __name__ == "__main__":
    app.run('0.0.0.0', port=cfg.DASHBOARD_PORT)
