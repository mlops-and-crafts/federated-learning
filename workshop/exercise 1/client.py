
from flwr.common import NDArrays, Scalar
from sklearn.linear_model import LinearRegression
import numpy as np
import flwr as fl
from sklearn.datasets import fetch_california_housing
from workshop.client.client import partition
from workshop.client.utils import set_initial_params

from typing import Dict, List, Tuple


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
        
    def get_parameters(self, config) -> List:
        """Returns the paramters of a sklearn LinearRegression model."""

    def fit(self, parameters, config) -> tuple[NDArrays, int, dict]:
        """Refit the model locally with the central parameters and return them"""
        updated_parameters = parameters
        num_examples = 0
        metrics = {} # Won't be used in this example, we can return it empty

        return updated_parameters, num_examples, metrics
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        mse = 0
        num_examples = 0
        metrics = {}
        

        return mse, num_examples, metrics
