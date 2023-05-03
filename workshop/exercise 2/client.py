from flwr.common import NDArrays, Scalar
from sklearn.linear_model import LinearRegression
import numpy as np
import flwr as fl
from sklearn.datasets import fetch_california_housing
from utils import partition, set_initial_params
from typing import Dict, List, Tuple
import logging
import time

'''
There are two categories of evaluation and monitoring in federated learning: server-side and client-side.
In general monitoring in different types of drift is very difficult for federated learning, because
the data stays on the device. You can construct metrics that can be sent to the server, but care
needs to be taken not to construct numbers that might give away too much information about the data.

To keep things simple for our implementation, we will only be passing the R-squared back to the server, in
addition to the loss and number of examples.

def evaluate(self, parameters, config): Evaluate the provided parameters using the locally held dataset.
Returns these outputs:
    loss (float) - The evaluation loss of the model on the local dataset.
    num_examples (int) - The number of examples used for evaluation.
    metrics (Dict[str, Scalar]) - A dictionary mapping arbitrary string keys to values of type bool, bytes, float, int, or str.
    It can be used to communicate arbitrary values back to the server.
'''


class CaliforniaHousingClient(fl.client.NumPyClient):
    def __init__(self, partition_id: int):
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

    def get_parameters(self, config) -> List:
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
            server_address = "0.0.0.0:8080"  # pick up the server address from system arguments

            client = CaliforniaHousingClient(partition_id=2)
            fl.client.start_numpy_client(
                server_address=server_address, client=client)
            break
        except Exception as e:
            logging.warning(
                "Could not connect to server: sleeping for 5 seconds...")
            time.sleep(5)