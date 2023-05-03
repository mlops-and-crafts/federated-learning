import flwr as fl
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import warnings
import logging
import time
import os
from utils import set_model_params, set_initial_params

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions),
            np.array_split(y, num_partitions))
    )


class CaliforniaHousingClient(fl.client.NumPyClient):
    def __init__(self, partition_id: int):
        self.data = None
        self.target = None
        self.model = LinearRegression()
        set_initial_params(self.model)

        X, y = fetch_california_housing(return_X_y=True)

        partition_id = np.random.choice(10)

        self.X_train, self.y_train = X[:15000], y[:15000]
        self.X_test, self.y_test = X[15000:], y[15000:]

        self.X_train, self.y_train = partition(
            self.X_train, self.y_train, 10)[partition_id]

    def get_parameters(self, config):
        """Returns the paramters of a sklearn LinearRegression model."""
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

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = self.model.fit(self.X_train, self.y_train)

        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        mse = 0.0
        num_examples = 100
        r_squared = 0
        # Make sure to leave the key name as r-squared
        metrics = {"r_squared": r_squared}
        return mse, num_examples, metrics


if __name__ == "__main__":
    while True:
        try:
            # pick up Ip from the os environment or pass them as sys args
            server_address = os.environ['SERVER_ADDRESS']
            server_port = os.environ["SERVER_PORT"]

            client = CaliforniaHousingClient(partition_id=2)
            fl.client.start_numpy_client(
                server_address=server_address + ":" + server_port,  client=client)
            break
        except Exception as e:
            logging.exception(e)
            logging.warning(
                "Could not connect to server: sleeping for 5 seconds...")
            time.sleep(5)
