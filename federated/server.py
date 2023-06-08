import flwr as fl
import logging
import time
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

from clustered_data import ClusteredDataGenerator
import server_config


logger = logging.getLogger("flwr")


def fit_metrics_aggregation_fn(metrics):
    clients_string = ""
    for i, client_metrics in enumerate(metrics):
        clients_string += " " + client_metrics[1]["client_name"]
        if i % 5 == 0 and i != 0:
            clients_string += "\n"

    logger.info("Connected client names:" + clients_string)
    logger.info(
        f"Sleeping for {server_config.SLEEP_TIME_BETWEEN_ROUNDS} seconds between rounds"
    )

    time.sleep(server_config.SLEEP_TIME_BETWEEN_ROUNDS)

    return {"connected_clients": clients_string}


def evaluate_fn(server_round, parameters, config):
    model.set_params(**config)
    model.intercept_ = parameters[0]
    model.coef_ = parameters[1]

    rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
    r_squared = model.score(X_test, y_test)
    metrics_dict = {"rmse": rmse, "r_squared": r_squared}
    logger.info(
        f"Round {server_round} RMSE: {rmse} R^2: {r_squared} coefs = {parameters}"
    )
    return rmse, metrics_dict


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    # X, y = fetch_california_housing(return_X_y=True)
    X, y, coef = make_regression(
        n_samples=20_000,
        n_features=5,
        bias=-2.0,
        n_informative=3,
        noise=1,
        random_state=42,
        coef=True,
    )
    # logger.info(f"TRUE COEFFICIENTS: {true_coef}")
    ClusteredDataset = ClusteredDataGenerator(
        X, y, n_clusters=50, test_size=0.2, seed=42
    )
    X_train, X_test, y_train, y_test = ClusteredDataset.get_train_test_data()
    model = SGDRegressor()
    while True:
        try:
            strategy = fl.server.strategy.FedAvg(
                min_available_clients=server_config.MIN_CLIENTS,
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_fn=evaluate_fn,
            )
            fl.server.start_server(
                server_address=f"0.0.0.0:{server_config.PORT}",
                strategy=strategy,
                config=fl.server.ServerConfig(
                    num_rounds=server_config.NUM_ROUNDS,
                    round_timeout=server_config.ROUND_TIMEOUT,
                ),
            )

        except Exception as e:
            logging.exception(e)
