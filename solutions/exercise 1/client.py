import flwr as fl
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import logging
import time
import os
from utils import set_initial_params, partition

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


np.random.seed(42)


class CaliforniaHousingClient(fl.client.NumPyClient):
    def __init__(self):
        self.data = None
        self.target = None
        self.model = LinearRegression()
        set_initial_params(self.model)

        X, y = fetch_california_housing(return_X_y=True)

        partition_id = np.random.choice(50)

        self.X_train, self.y_train = X[:15000], y[:15000]
        self.X_test, self.y_test = X[15000:], y[15000:]

        self.X_train, self.y_train = partition(self.X_train, self.y_train, 50)[
            partition_id
        ]

    def set_model_params(self, params):
        """Sets the parameters of a sklean Regression model."""
        self.model.coef_ = params[0]
        if self.model.fit_intercept:
            self.model.intercept_ = params[1]

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
        self.set_model_params(parameters)

        self.model = self.model.fit(self.X_train, self.y_train)

        return self.get_parameters(config), len(self.X_train), {"client_name": "Pietje"}

    def evaluate(self, parameters, config):
        mse = 0.0
        num_examples = 100
        dummy = 0
        # Make sure to leave the key name as r-squared
        metrics = {"dummy": dummy}
        return mse, num_examples, metrics


if __name__ == "__main__":
    while True:
        try:
            # pick up Ip from the os environment or pass them as sys args
            server_address = os.environ["SERVER_ADDRESS"]
            server_port = os.environ["SERVER_PORT"]

            client = CaliforniaHousingClient()
            fl.client.start_numpy_client(
                server_address=server_address + ":" + server_port, client=client
            )
        except Exception as e:
            logging.exception(e)
            logging.warning("Could not connect to server: sleeping for 5 seconds...")
        
        time.sleep(10)
