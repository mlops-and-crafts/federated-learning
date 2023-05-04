from flwr.common import NDArrays, Scalar
from sklearn.linear_model import LinearRegression
import numpy as np
import flwr as fl
from sklearn.datasets import fetch_california_housing
from utils import partition, set_initial_params
from typing import Dict, Tuple
import logging
import time
import urllib

'''
Flower has a built-in functionality to secure communcation with the server through SSL encryption.

The exercise here is to pass the root certificate to the client.
'''


class CaliforniaHousingClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = LinearRegression()

        # At the beginning of the run the server requests the parameters of the model.
        # Set these to 0 so that it does not return the None values for an untrained model
        set_initial_params(self.model)

        X, y = fetch_california_housing(return_X_y=True)

        partition_id = np.random.choice(10)

        self.X_train, self.y_train = X[:15000], y[:15000]
        self.X_test, self.y_test = X[15000:], y[15000:]

        self.X_train, self.y_train = partition(
            self.X_train, self.y_train, 10)[partition_id]

    def get_parameters(self, config) -> NDArrays:
        """Returns the paramters of a sklearn LinearRegression model."""

        parameters = []

        return parameters

    def fit(self, parameters, config) -> tuple[NDArrays, int, dict]:
        """Refit the model locally with the central parameters and return them"""
        updated_parameters = parameters
        num_examples = 0
        metrics = {}  # Won't be used in this example, we can return it empty

        return updated_parameters, num_examples, metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        '''
        You can leave this method as is for now.
        '''
        mse = 20
        num_examples = 100
        metrics = {"dummy": 0}

        return mse, num_examples, metrics


if __name__ == "__main__":
    while True:
        try:
            # pick up IP and port from the os environment or pass them as sys args
            server_address = ""
            server_port = ""

            # Download the root certificate from here "http://{server_address}:8000/ca.crt" and store it safely
            # treat it like a secret! You can use urllib to directly read and pass it, or you can download it,
            # store it somehwere safe and read it in.

            client = CaliforniaHousingClient()
            fl.client.start_numpy_client(
                server_address=server_address + ":" + server_port, client=client, root_certificates="")
            break
        except Exception as e:
            logging.warning(
                "Could not connect to server: sleeping for 5 seconds...")
            time.sleep(5)
