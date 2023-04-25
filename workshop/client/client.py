import flwr as fl
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
import logging
import time
from pathlib import Path

from server_src.utils import set_model_params, set_initial_params

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions),
            np.array_split(y, num_partitions))
    )


class CaliforniaHousingClient(fl.client.NumPyClient):
    def __init__(self):
        self.data = None
        self.target = None
        self.model = LinearRegression()
        set_initial_params(self.model)

        X, y = fetch_california_housing(return_X_y=True)

        partition_id = np.random.choice(10)

        self.X_train, self.y_train = X[:15000], y[:15000]
        self.X_test, self.y_test = X[15000:], y[15000:]

        partition_id = np.random.choice(10)
        self.X_train, self.y_train = partition(
            self.X_train, self.y_train, 10)[partition_id]

    def get_parameters(self, config):
        """Returns the paramters of a sklearn LogisticRegression model."""
        if self.model.fit_intercept:
            params = [
                self.model.coef_,
                self.model.intercept_,
            ]
        else:
            params = [
                self.model.coef_,
            ]
        return params

    def fit(self, parameters, config):  # type: ignore
        set_model_params(self.model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = self.model.fit(self.X_train, self.y_train)
        print(f"Training finished for round {config['server_round']}")
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        set_model_params(self.model, parameters)
        mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        r_squared = self.model.score(self.X_test, self.y_test)
        return mse, len(self.X_test), {"r_squared": r_squared}


if __name__ == "__main__":
    while True:
        try:
            client = CaliforniaHousingClient()
            fl.client.start_numpy_client(
                server_address="localhost:5040", client=client, root_certificates=Path(".cache/certificates/ca.crt").read_bytes())
            break
        except Exception as e:
            logging.warning(
                "Could not connect to server: sleeping for 5 seconds...")
            time.sleep(5)
