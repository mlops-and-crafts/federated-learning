import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import flwr as fl
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error

from clustered_data import ClusteredScaledDataGenerator
import cfg
from metrics import MetricsJSONstore

logger = logging.getLogger("flwr")
filehandler = logging.FileHandler(f'{cfg.LOGFILE_DIR}/server.log', mode='w')
filehandler.setLevel(logging.INFO)
logger.addHandler(filehandler)

metrics = MetricsJSONstore(cfg.METRICS_FILE)
            

def fit_metrics_aggregation_fn(metrics):
    """
    Aggregates the metrics returned by the clients after a round of training.
    In our case we simply aggregate the names of the clients into a single string and log it.
    Then we sleep for a few seconds until the next round of federating learning. 

    Args:
        metrics (List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]): A list of tuples containing the
        client proxy and the fit results returned by the client.

    Returns:
        dict: A dictionary containing the connected clients' names that finished FIT this round.
    """
    clients_string = ""
    for i, client_fit_metrics in enumerate(metrics):
        clients_string += " " + client_fit_metrics[1]["client_name"]
        if i % 5 == 0 and i != 0:
            clients_string += "\n"

    logger.info("Connected client names that finished FIT this round:" + clients_string)
    logger.info(f"Sleeping for {cfg.SLEEP_TIME_BETWEEN_ROUNDS} seconds between rounds")
    time.sleep(cfg.SLEEP_TIME_BETWEEN_ROUNDS)

    return {"connected_clients": clients_string}


def evaluate_metrics_aggregation_fn(metrics):
    """
    Aggregates the metrics returned by the clients after a round of evaluation.
    The function calculates the average and standard deviation of the RMSE (root mean squared error) values
    returned by the clients, and logs the connected client names, the average RMSE, and the standard deviation
    of the RMSE.

    Args:
        metrics (List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]]): A list of tuples containing
        the client proxy and the evaluation results returned by the client.

    Returns:
        dict: A dictionary containing the connected clients' names that finished EVALUATE this round, the average RMSE,
        and the standard deviation of the RMSE.
    """
    clients_string = ""
    client_rmses = []
    for i, client_metrics in enumerate(metrics):
        clients_string += " " + client_metrics[1]["client_name"]
        if i % 5 == 0 and i != 0:
            clients_string += "\n"
        client_rmses.append(client_metrics[1]["rmse"])

    avg_rmse = np.mean(client_rmses)
    std_rmse = np.std(client_rmses)
    logger.info(
        "Connected client names that finished EVALUATE this round:" + clients_string
    )
    logger.info(
        f"AVG client RMSE: {avg_rmse:.2f} (central={central_rmse}), "
        f"STD client RMSE: {std_rmse:.2f}"
    )


    return {
        "connected_clients": clients_string,
        "avg_rmse": avg_rmse,
        "std_rmse": std_rmse,
    }


def evaluate_fn(server_round, parameters, config):
    """
    Evaluates the federated model using the given parameters and configuration.

    Args:
        server_round (int): The current server round.
        parameters (list): A list containing the intercept and coefficients of the model.
        config (dict): A dictionary containing the configuration of the model.

    Returns:
        tuple: A tuple containing the federated RMSE and a dictionary of metrics.
    """
    federated_model.set_params(**config)
    federated_model.intercept_ = parameters[0]
    federated_model.coef_ = parameters[1]

    federated_rmse = mean_squared_error(
        y_test.values, federated_model.predict(X_test), squared=False
    )
    federated_r_squared = federated_model.score(X_test, y_test)
    metrics_dict = {"server_round": server_round, "rmse": federated_rmse, "r_squared": federated_r_squared}
    logger.info(
        f"SERVER Round {server_round} RMSE: {federated_rmse} R^2: {federated_r_squared} coefs = {parameters}"
    )

    metrics.log_server_metrics(metrics_dict)
    return federated_rmse, metrics_dict



class CustomFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    A custom implementation of the FedAvg strategy that logs client fit and client evaluate results.

    Inherits from `fl.server.strategy.FedAvg`.

    Attributes:
        None

    Methods:
        aggregate_fit(server_round, results, failures):
            Aggregates the fit results from the clients and logs them.
        aggregate_evaluate(server_round, results, failures):
            Aggregates the fit results from the clients and logs them.

    Usage:
        Use this class to create a custom FedAvg strategy that logs client fit results.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.typing.Parameters], Dict[str, fl.common.typing.Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        # TODO: log client fit results
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.typing.Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        # TODO: log client evaluate results
        return loss_aggregated, metrics_aggregated


if __name__ == "__main__":
    if cfg.USE_HOUSING_DATA:
        X, y = fetch_california_housing(return_X_y=True)
    else:
        X, y, coef = make_regression(
            n_samples=20_000,
            n_features=5,
            bias=-2.0,
            n_informative=3,
            noise=1,
            random_state=42,
            coef=True,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        logger.info(f"TRUE COEFFICIENTS: {coef}")

    X_train, X_test, y_train, y_test = ClusteredScaledDataGenerator(
        pd.DataFrame(X),
        pd.Series(y),
        n_clusters=cfg.N_CLUSTERS,
        test_size=cfg.TEST_SIZE,
        seed=cfg.SEED,
    ).get_train_test_data()

    logger.debug(
        f"SERVER X_train shape: {X_train.shape} y_train shape: {y_train.shape}"
    )
    logger.debug(f"SERVER X_test shape: {X_test.shape} y_est shape: {y_test.shape}")

    central_model = LinearRegression().fit(X_train, y_train)
    central_rmse = mean_squared_error(
        y_test, central_model.predict(X_test), squared=False
    )
    central_r_squared = central_model.score(X_test, y_test)
    logger.info(f"SERVER Central RMSE: {central_rmse} R^2: {central_r_squared}")

    federated_model = SGDRegressor()
    while True:
        try:
            strategy = CustomFedAvgStrategy(
                min_available_clients=cfg.MIN_CLIENTS,
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_fn=evaluate_fn,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            )

            fl.server.start_server(
                server_address=f"0.0.0.0:{cfg.SERVER_PORT}",
                strategy=strategy,
                config=fl.server.ServerConfig(
                    num_rounds=cfg.NUM_ROUNDS,
                    round_timeout=cfg.ROUND_TIMEOUT,
                ),
            )
        except Exception as e:
            logging.exception(e)
