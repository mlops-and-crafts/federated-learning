import flwr as fl
import logging
import time
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

logger = logging.getLogger("flwr")


def fit_metrics_aggregation_fn(metrics):
    clients_string = ""
    for i, client_metrics in enumerate(metrics):
        clients_string += " " + client_metrics[1]["client_name"]
        if i % 5 == 0 and i != 0:
            clients_string += "\n"

    logger.info("Connected client names:" + clients_string)
    logger.info("Sleeping for 10 seconds between rounds")

    time.sleep(10)


def get_evaluate_fn(model: LinearRegression):
    """Return an evaluation function for server-side evaluation."""
    X, y = fetch_california_housing(return_X_y=True)
    X_test, y_test = X[15000:], y[15000:]

    def evaluate(server_round, parameters, config):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]

        loss = mean_squared_error(y_test, model.predict(X_test))
        r_squared = model.score(X_test, y_test)
        return loss, {"r_squared": r_squared}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LinearRegression()
    while True:
        try:
            strategy = fl.server.strategy.FedAvg(
                min_available_clients=2,
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_fn=get_evaluate_fn(model),
            )
            fl.server.start_server(
                server_address="0.0.0.0:8080",
                strategy=strategy,
                config=fl.server.ServerConfig(num_rounds=2000000),
            )
        except Exception as e:
            logging.exception(e)

        logger.info("Sleeping for 10 seconds before next round.")
