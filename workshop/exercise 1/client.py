from flwr.common import NDArrays, Scalar
from sklearn.linear_model import LinearRegression
import numpy as np
import flwr as fl
from sklearn.datasets import fetch_california_housing
from utils import partition, set_initial_params
from typing import Dict, Tuple
import logging
import time
import os

"""
Imagine you're an engineer tasked with federating a model provided to you by a data scientist.
In this exercise we will be exploring the basic functionality of the flower library to achieve
this goal.

We will complete the following steps:

    1.  Write a basic client
    2.  Dockerize the client
    3.  Deploy and run federated learning rounds with (hopefully) all workshop participants!

The client is flower's abstract class that executes the code on the edge. It runs code to
train the local model, updates the model
parameters from parameters received from the server and runs evaluations on local data. Because
the client is an abstract class, some methods
do not have a default implementation. These methods are the methods that define how the local model is trained
and how parameter updates get sent from the edge to the server.

def get_parameters(self, config) -> NDArrays (list of mutable iterables): defines how parameters are extracted from the
edge model and how they are sent back to the server.
Our server already exists, and expects parameters in a two-dimensional list format,
with the format being [coefficients (NDArrays), intercept (NDArrays)].
The config parameter is mandatory in this method, but we will not use it in our implementation, therefore it is empty.


def fit(self, parameters, config): -> List, Integer, Dict defines how the model will be refit, using the parameters passed by the server.
The parameters argument is passed by the global (server) model. Configuration parameters allow the server to influence training on the client.
It can be used to communicate arbitrary values from the server to the client, for example, to set the number of (local) training epochs.
It returns the following three outputs:

    parameters (NDArrays) - The locally updated model parameters.
    num_examples (int) - The number of examples used for training.
    metrics (Dict[str, Scalar]) - A dictionary mapping arbitrary string keys to values of type bool, bytes,
    float, int, or str. It can be used to communicate arbitrary values back to the server.

We have already written a client class with the skeleton methods below. Your task is to populate these methods.
When you're finished, populate the dockerfile with the IP address and port that will be provided to you and pass them to the file entrypoint (if __name__ = "__main__"). 

When you're done, feel free to launch your client to check whether it connects to the server!
You can do that by building the docker image and drunning it.

docker build -t client PATH_TO_CLIENT
docker run client
"""


class CaliforniaHousingClient(fl.client.NumPyClient):
    def __init__(self):
        # Initilialise the model
        self.model = LinearRegression()

        # At the beginning of the run the server requests the parameters of the model.
        # Set these to 0 so that it does not return the None values for an untrained model
        set_initial_params(self.model)

        X, y = fetch_california_housing(return_X_y=True)

        # Define the data and partition it randomly across "devices"
        # Select a sample of the original dataset in order to mimic a real life situation in which every edge device has a different dataset
        partition_id = np.random.choice(50)

        self.X_train, self.y_train = X[:15000], y[:15000]
        self.X_test, self.y_test = X[15000:], y[15000:]

        self.X_train, self.y_train = partition(self.X_train, self.y_train, 50)[
            partition_id
        ]

    def get_parameters(self, config) -> NDArrays:
        """Returns the paramters of a sklearn LinearRegression model."""

        parameters = []

        return parameters

    def fit(self, parameters, config) -> tuple[NDArrays, int, dict]:
        """Refit the model locally with the central parameters and return them"""
        # 1. Set model params from global model
        # 2. Refit model on local data

        updated_parameters = parameters
        num_examples = 0
        # change the value to your name to see whether your client is connecting to the server correctly
        metrics = {"client_name": "client"}

        return updated_parameters, num_examples, metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        You can leave this method untouched for now.
        """
        mse = 20
        num_examples = 100
        metrics = {"dummy": 0}

        return mse, num_examples, metrics


if __name__ == "__main__":
    while True:
        try:
            # pass IP and port from the os environment or pass them as sys args
            server_address = ""
            server_port = ""

            client = CaliforniaHousingClient()
            fl.client.start_numpy_client(
                server_address=server_address + ":" + server_port, client=client
            )
            break
        except Exception as e:
            logging.warning("Could not connect to server: sleeping for 5 seconds...")
            time.sleep(5)
