
import logging
import time
import os
import ssl
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import client_config
from clustered_data import ClusteredDataGenerator

logging.basicConfig(level=logging.DEBUG)
ssl._create_default_https_context = ssl._create_unverified_context


class SGDRegressorClient(fl.client.NumPyClient):
    def __init__(self, X_train, X_test, y_train, y_test, name="client"):
        self.name  = name
        self.edge_model = SGDRegressor()
        self.federated_model = SGDRegressor()

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # partition_id = np.random.choice(50)

        # #X, y = fetch_california_housing(return_X_y=True)
        # X, y, coef = make_regression(n_samples=20_000, n_features=5, bias=-2.0, n_informative=3, noise=1, random_state=42, coef=True)

        # self.X_train, self.y_train = X[:15000], y[:15000]
        # self.X_test, self.y_test = X[15000:], y[15000:]
        # N_PARTITIONS = 50
        # self.X_train, self.y_train = self._partition_dataset(self.X_train, self.y_train, N_PARTITIONS)[
        #     partition_id
        # ]

        self.n_features = len(self.X_train[0])
        self._set_model_zero_coefs(self.edge_model, self.n_features)
        self._set_model_zero_coefs(self.federated_model, self.n_features)

        logging.debug(
            f"Initialized {self.name} with "
            f"X_train.shape={self.X_train.shape} X_test.shape={self.X_test.shape}..."
        )
    
    def _set_model_zero_coefs(self, model:SGDRegressor, n_features:int)->None:
        model.intercept_ = np.zeros((1,))
        model.coef_ = np.zeros((n_features,))
        
    def _set_model_coefs(self, model:SGDRegressor, params:List[np.ndarray])->None:
        """Sets the parameters of a sklean Regression model."""
        model.intercept_ = params[0]
        model.coef_ = params[1]

    def _get_model_coefs(self, model:SGDRegressor) -> List[np.ndarray]:
        """Returns the paramters of a sklearn LinearRegression model."""
        coefs = [model.intercept_, model.coef_]
        return coefs
    
    def get_parameters(self, config):
        params = self._get_model_coefs(self.federated_model)
        logging.debug(f"Client {self.name} sending parameters: {params}")
        return params
    
    def set_parameters(self, parameters, config):
        logging.debug(f"Client {self.name} received parameters {parameters} and {config}")
        self.federated_model.set_params(**config)
        self._set_model_coefs(self.federated_model, parameters)
        

    def fit(self, parameters, config):
        self.edge_model.partial_fit(self.X_train, self.y_train)
        self.federated_model.set_params(**config)
        self.federated_model.partial_fit(self.X_train, self.y_train)

        federated_model_coefs = self._get_model_coefs(self.federated_model)
        n_samples = len(self.X_train)
        metadata = {"client_name": self.name}

        logging.info(f"Client {self.name} fit model with {n_samples} samples.")
        return federated_model_coefs, n_samples, metadata
    
    def _get_rmse(self, model:SGDRegressor) -> float:
        return mean_squared_error(self.y_test, model.predict(self.X_test), squared=False)

    def evaluate(self, parameters, config):
        edge_rmse = self._get_rmse(self.edge_model)
        federated_rmse = self._get_rmse(self.federated_model)

        central_model = SGDRegressor()
        central_model.set_params(**config)
        self._set_model_coefs(central_model, parameters)
        
        central_rmse = self._get_rmse(central_model)
        n_samples = len(self.X_test)
        # Make sure to leave the key name as r-squared
        metrics = {"rmse": central_rmse}
        logging.info(f"Client {self.name} evaluated rmse: edge={edge_rmse} federated={federated_rmse} central={central_rmse}...")
        return central_rmse, n_samples, metrics


if __name__ == "__main__":
    time.sleep(1) # wait for server to start
    
    #X, y = fetch_california_housing(return_X_y=True)
    X, y, coef = make_regression(n_samples=20_000, n_features=5, bias=-2.0, n_informative=3, noise=1, random_state=42, coef=True)
    ClusteredDataset = ClusteredDataGenerator(X, y, n_clusters=50, test_size=0.2, seed=42)
    X_train, X_test, y_train, y_test = ClusteredDataset.get_random_cluster_train_test_data()
    
    client = SGDRegressorClient(X_train, X_test, y_train, y_test)
    server_address = f"{client_config.SERVER_ADDRESS}:{client_config.SERVER_PORT}"
    
    while True:
        try:
            fl.client.start_numpy_client(server_address=server_address, client=client)
        except Exception as e:
            logging.exception(e)
            logging.warning(f"Could not connect to server: sleeping for {client_config.RETRY_SLEEP_TIME} seconds...")

        time.sleep(client_config.RETRY_SLEEP_TIME)
