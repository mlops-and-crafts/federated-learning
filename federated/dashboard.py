import logging

import pandas as pd

from dash import Dash, dcc, no_update, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


import plotly.express as px
import plotly.graph_objs as go

import cfg
from helpers import MetricsJSONstore

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("federated-dashboard")

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ, dbc_css])
app.title = "Federated Learning Dashboard"
app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            brand="MLOps and Crafts: Federated Learning Dashboard",
            color="primary",
            brand_href="https://www.meetup.com/nl-NL/mlops-and-crafts/",
            style={"margin-bottom": 15},
        ),
        dbc.Card(
            [
                dbc.CardHeader("Configuration"),
                dbc.CardBody(
                    [
                        dash_table.DataTable(
                            id="config-table",
                            data=[
                                {
                                    "USE_HOUSING_DATA": cfg.USE_HOUSING_DATA,
                                    "CLUSTER_METHOD": cfg.CLUSTER_METHOD,
                                }
                            ],
                            columns=[
                                {"name": col, "id": col}
                                for col in ["USE_HOUSING_DATA", "CLUSTER_METHOD"]
                            ],
                        ),
                    ],
                    className="dbc",
                ),
            ],
            style={"margin-bottom": 15},
        ),
        dbc.Card(
            [
                dbc.CardHeader("Federated vs Centralized model comparison"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(id="federated-rmse-graph"),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(id="federated-coefs-graph"),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            style={"margin-bottom": 15},
        ),
        dbc.Card(
            [
                dbc.CardHeader("Federated vs Edge local performance comparison"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(id="client-federated-rmse-graph"),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(id="client-edge-rmse-graph"),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            style={"margin-bottom": 15},
        ),
        dbc.Card(
            [
                dbc.CardHeader("Client metrics"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dash_table.DataTable(
                                            id="latest-client-metrics-table",
                                        ),
                                    ],
                                    className="dbc",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label(
                                            "Select client to compare edge vs federated performance",
                                            html_for="client-dropdown",
                                        ),
                                        dbc.Select(
                                            id="client-dropdown",
                                            placeholder="Select client...",
                                        ),
                                        dcc.Graph(id="client-rmse-comparison-graph"),
                                    ],
                                    style={"margin-top": 15},
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            style={"margin-bottom": 15},
        ),
        dcc.Interval(
            id="interval-ticker",
            interval=1000,  # update and reload json metrics file every second
            n_intervals=0,
        ),
        dcc.Store(id="metrics-store"),
    ]
)


@app.callback(Output("metrics-store", "data"), Input("interval-ticker", "n_intervals"))
def update_metrics(n_intervals):
    metrics = MetricsJSONstore(cfg.METRICS_FILE).load()
    logger.debug("Latest metrics loaded from file")
    return metrics


@app.callback(
    Output("client-dropdown", "options"),
    Output("client-dropdown", "value"),
    Input("metrics-store", "data"),
    State("client-dropdown", "value"),
)
def update_client_dropdown(metrics, client):
    if metrics is None:
        return [], no_update
    client_names = [
        {"label": client_name, "value": client_name}
        for client_name in set(
            [client["client_name"] for client in metrics["clients_evaluate"]]
        )
    ]
    if not client:
        return client_names, client_names[0]
    return client_names, no_update


@app.callback(
    Output("federated-rmse-graph", "figure"),
    Input("metrics-store", "data"),
)
def updated_federated_graph(metrics):
    try:
        server_metrics_df = pd.DataFrame(metrics["server"])
        client_fit_metrics_df = pd.DataFrame(metrics["clients_fit"])

        if not server_metrics_df.empty:
            server_federated_fig = go.Figure()
            server_federated_fig.add_traces(
                [
                    go.Scatter(
                        x=server_metrics_df["server_round"].values,
                        y=server_metrics_df["rmse"].values,
                        name="Federated Model",
                        line=dict(width=4),
                        mode="lines",
                    ),
                    go.Scatter(
                        x=server_metrics_df["server_round"].values,
                        y=server_metrics_df["central_rmse"].values,
                        name="Central Model Baseline",
                        line=dict(width=2),
                        mode="lines",
                    ),
                ]
            )

            for client in client_fit_metrics_df["client_name"].unique():
                sub_df = client_fit_metrics_df[
                    client_fit_metrics_df["client_name"] == client
                ]
                server_federated_fig.add_trace(
                    go.Scatter(
                        x=sub_df["server_round"].values,
                        y=sub_df["client_rmse"].values,
                        name=client,
                        line=dict(dash="dash"),
                        mode="lines",
                    )
                )
            server_federated_fig.update_layout(
                title="federated RMSE against central test set"
            )
            return server_federated_fig
        else:
            raise Exception("Server metrics dataframe is empty")
    except Exception as e:
        logger.exception("Failed to display federated graph in dashboard", exc_info=e)
        raise PreventUpdate


@app.callback(
    Output("federated-coefs-graph", "figure"),
    Input("metrics-store", "data"),
)
def updated_federated_coefs_graph(metrics):
    try:
        server_metrics_df = pd.DataFrame(metrics["server"])

        if not server_metrics_df.empty:
            server_metrics_df.sort_values("server_round", inplace=True)
            federated_coefs = server_metrics_df.iloc[-1]["coefs"]
            central_coefs = server_metrics_df.iloc[-1]["central_coefs"]

            fig = go.Figure()
            fig.add_traces(
                [
                    go.Bar(
                        x=list(federated_coefs.keys()),
                        y=list(federated_coefs.values()),
                        name="Federated Model",
                    ),
                    go.Bar(
                        x=list(central_coefs.keys()),
                        y=list(central_coefs.values()),
                        name="Central Model",
                    ),
                ]
            )
            fig.update_layout(title="Federated vs Central Model Coefficients")
            return fig
        else:
            raise PreventUpdate
    except Exception as e:
        logger.exception(
            "Failed to display federated coefficient graph in dashboard", exc_info=e
        )
        raise PreventUpdate


@app.callback(
    Output("client-federated-rmse-graph", "figure"),
    Output("client-edge-rmse-graph", "figure"),
    Output("latest-client-metrics-table", "data"),
    Input("metrics-store", "data"),
)
def update_client_graphs_and_table(metrics):
    try:
        client_eval_metrics_df = pd.DataFrame(metrics["clients_evaluate"])

        client_federated_fig = (
            px.line(
                client_eval_metrics_df,
                x="server_round",
                y="federated_rmse",
                title="Federated RMSE against local test data for client models",
                color="client_name",
            )
            if not client_eval_metrics_df.empty
            else no_update
        )
        client_edge_fig = (
            px.line(
                client_eval_metrics_df,
                x="server_round",
                y="edge_rmse",
                title="Edge RMSE against local test data for client models",
                color="client_name",
            )
            if not client_eval_metrics_df.empty
            else no_update
        )

        last_client_update_df = (
            (
                client_eval_metrics_df[
                    client_eval_metrics_df.server_round
                    == client_eval_metrics_df.server_round.max()
                ].sort_values("client_name")
            )
            if not client_eval_metrics_df.empty
            else pd.DataFrame()
        )
        return (
            client_federated_fig,
            client_edge_fig,
            last_client_update_df.to_dict("records"),
        )
    except Exception as e:
        logger.exception("Failed to display client graphs in dashboard", exc_info=e)
        raise PreventUpdate


@app.callback(
    Output("client-rmse-comparison-graph", "figure"),
    Input("metrics-store", "data"),
    State("client-dropdown", "value"),
)
def update_client_rmse_comparison(metrics, client):
    if client is not None:
        try:
            client_eval_metrics_df = pd.DataFrame(metrics["clients_evaluate"])

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=client_eval_metrics_df[
                        client_eval_metrics_df["client_name"] == client
                    ]["server_round"].values,
                    y=client_eval_metrics_df[
                        client_eval_metrics_df["client_name"] == client
                    ]["federated_rmse"].values,
                    name="Federated RMSE",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=client_eval_metrics_df[
                        client_eval_metrics_df["client_name"] == client
                    ]["server_round"].values,
                    y=client_eval_metrics_df[
                        client_eval_metrics_df["client_name"] == client
                    ]["edge_rmse"].values,
                    name="Edge RMSE",
                )
            )
            fig.update_layout(title=f"RMSE against local test data for {client}")
            return fig
        except Exception as e:
            logger.exception(
                "Failed to display client comparison graphs in dashboard", exc_info=e
            )
    raise PreventUpdate


if __name__ == "__main__":
    app.run("0.0.0.0", port=cfg.DASHBOARD_PORT)
