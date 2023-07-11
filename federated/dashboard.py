import re
from pathlib import Path
from typing import Union, List, Dict
import logging

import numpy as np
import pandas as pd

from dash import Dash, html, dcc, no_update, dash_table
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

    html.Div([
        dcc.Graph(id='client-federated-rmse-graph'),
    ]),
    html.Div([
        dcc.Graph(id='client-edge-rmse-graph'),
    ]),
    html.Div([
        dash_table.DataTable(id='latest-client-metrics-table'),
    ]),
    dcc.Interval(
        id='interval-ticker',
        interval=1000, # in milliseconds
        n_intervals=0
    )
])

@app.callback(
    Output('federated-rmse-graph', 'figure'), 
    Output('client-federated-rmse-graph', 'figure'), 
    Output('client-edge-rmse-graph', 'figure'), 
    Output('latest-client-metrics-table', 'data'),
    Input('interval-ticker', 'n_intervals')
)
def update_data(n_intervals):
    try:
        metrics = MetricsJSONstore(cfg.METRICS_FILE).load()
        server_metrics_df = pd.DataFrame(metrics["server"])
        
        server_federated_fig = px.line(
            server_metrics_df, x="server_round", y="rmse", 
            title='RMSE against test data for federated model'
        )
        client_metrics_df = pd.DataFrame(metrics["clients_evaluate"])

        last_client_update_df = (
            client_metrics_df[client_metrics_df.server_round == client_metrics_df.server_round.max()]
        )
        
        client_federated_fig = px.line(
            client_metrics_df, x="server_round", y="federated_rmse", 
            title='Federated RMSE against test data for client models',
            color="client_name",
        )
        client_edge_fig = px.line(
            client_metrics_df, x="server_round", y="edge_rmse", 
            title='Edge RMSE against test data for client models',
            color="client_name",
        )
        return server_federated_fig, client_federated_fig, client_edge_fig, last_client_update_df.to_dict('records')    
    
    except Exception as e:
        logger.error("Failed to display rmse graphs in dashboard:", e)
        return no_update, no_update, no_update, no_update

if __name__ == "__main__":
    app.run('0.0.0.0', port=cfg.DASHBOARD_PORT)
