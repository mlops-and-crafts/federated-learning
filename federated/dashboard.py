import dash
from dash import html
from dash import dcc, dash_table
import numpy as np
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import re
import cfg


def parse_server_metrics_from_log(fp):
    result_dict = dict()
    with open(fp, 'r') as f:
        server_rmse_lines = [x for x in f.readlines() if 'SERVER Round' in x]
        for line in server_rmse_lines:
            m = re.match('SERVER Round (.*) RMSE: (.*) R\^2: (.*) coefs = .*', line)
            result_dict[int(m.group(1))] = {
                'RMSE': float(m.group(2)), 'R-squared': float(m.group(3))
                }
    result = pd.DataFrame.from_dict(result_dict, orient='index')
    return result

fp = f'{cfg.LOGFILE_DIR}/server.log'
app = dash.Dash(__name__, update_title=None)
loss_data = parse_server_metrics_from_log(fp)
figure = px.line(loss_data, x=loss_data.index, y="RMSE", title='Federated loss')
graph = dcc.Graph(id='live-update-graph', figure=figure)
update_interval =  dcc.Interval(
            id='interval-component1',
            interval=1000, # in milliseconds
            n_intervals=0
        )
div1 = html.Div([graph, update_interval])
app.layout = html.Div([div1])

@app.callback(
        Output('live-update-graph', component_property='figure'), 
        Input('interval-component1', 'n_intervals')
)
def update_data(n_intervals):
    data = parse_server_metrics_from_log(fp)
    figure = px.line(data, x=data.index, y="RMSE", title='Federated loss')
    return figure

if __name__ == "__main__":
    app.run('0.0.0.0', port=cfg.DASHBOARD_PORT)
