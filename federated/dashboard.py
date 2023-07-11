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

logger = logging.getLogger("federated-dashboard")


def parse_server_metrics_from_log(server_log_path:Union[str, Path]) -> pd.DataFrame:
    result_dict = dict()
    with open(Path(server_log_path), 'r') as f:
        server_rmse_lines = [x for x in f.readlines() if 'SERVER Round' in x]
        for line in server_rmse_lines:
            regexp_result = re.match('SERVER Round (.*) RMSE: (.*) R\^2: (.*) coefs = .*', line)
            result_dict[int(regexp_result.group(1))] = {
                'RMSE': float(regexp_result.group(2)), 
                'R-squared': float(regexp_result.group(3))
            }
    result_df = pd.DataFrame.from_dict(result_dict, orient='index')
    return result_df

server_log_path = Path(f'{cfg.LOGFILE_DIR}/server.log')

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
    if server_log_path.exists():
        metrics_df = parse_server_metrics_from_log(server_log_path)
        if "RMSE" in metrics_df.columns:
            figure = px.line(
                metrics_df, 
                x=metrics_df.index, 
                y="RMSE", 
                title='RMSE against test data for federated model'
            )
            return figure
    logger.info(f"Server log file {server_log_path} does not exist yet")
    return no_update

if __name__ == "__main__":
    app.run('0.0.0.0', port=cfg.DASHBOARD_PORT)
