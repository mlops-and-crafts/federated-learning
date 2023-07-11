import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.common import  parameters_to_ndarrays
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error

from helpers import ClusteredScaledDataGenerator, MetricsJSONstore, get_test_rmse_from_parameters
import cfg

logger = logging.getLogger("flwr")
filehandler = logging.FileHandler(f'{cfg.LOGFILE_DIR}/server.log', mode='w')
filehandler.setLevel(logging.INFO)
logger.addHandler(filehandler)

metrics = MetricsJSONstore(cfg.METRICS_FILE)
            

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
    federated_rmse = get_test_rmse_from_parameters(parameters, config, X_test, y_test)
    metrics_dict = {
        "server_round": server_round, 
        "coefs": parameters[1].tolist(),
        "rmse": federated_rmse, 
        "central_rmse": central_rmse,
        "central_coefs": central_coefs,
    }

    logger.info(f"SERVER Round {server_round} RMSE: {federated_rmse} coefs = {parameters}")
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
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.typing.Parameters], Dict[str, fl.common.typing.Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Log client fit metrics to json file:
        for _, result in results:
            client_fit_metrics = {
                "server_round": server_round,
                "client_name": result.metrics["client_name"],
                "n_samples": result.metrics["n_samples"],
                "client_rmse": get_test_rmse_from_parameters(
                    parameters_to_ndarrays(result.parameters), {}, X_test, y_test,
                )
            }
            metrics.log_client_fit_metrics(client_fit_metrics)
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.typing.Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        # Log client evaluate metrics to json file:
        for _, result in results:
            client_evaluate_metrics = {
                "server_round": server_round,
                "client_name": result.metrics["client_name"],
                "n_samples": result.metrics["n_samples"],
                "edge_rmse": result.metrics["edge_rmse"],
                "federated_rmse": result.metrics["federated_rmse"],
                "central_rmse": result.metrics["central_rmse"],
                
            }
            metrics.log_client_evaluate_metrics(client_evaluate_metrics)
        time.sleep(cfg.SLEEP_TIME_BETWEEN_ROUNDS)
        return loss_aggregated, metrics_aggregated


if __name__ == "__main__":
    if cfg.USE_HOUSING_DATA:
        X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    else:
        X, y, true_coefs = make_regression(
            n_samples=20_000,
            n_features=5,
            bias=-2.0,
            n_informative=3,
            noise=1,
            random_state=42,
            coef=True,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        logger.info(f"TRUE COEFFICIENTS: {true_coefs}")

    X_train, X_test, y_train, y_test = ClusteredScaledDataGenerator(
        X, y,
        test_size=cfg.TEST_SIZE,
        n_clusters=cfg.N_CLUSTERS,
        seed=cfg.SEED,
    ).get_train_test_data()

    central_model = LinearRegression().fit(X_train, y_train)
    central_rmse = mean_squared_error(
        y_test, central_model.predict(X_test), squared=False
    )
    central_coefs = central_model.coef_.tolist()

    federated_model = SGDRegressor()
    federated_model.feature_names_in_ = np.array(X_train.columns, )
    while True:
        try:
            strategy = CustomFedAvgStrategy(
                min_available_clients=cfg.MIN_CLIENTS,
                evaluate_fn=evaluate_fn,
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
