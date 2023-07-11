import logging
import time
import os
import ssl
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import cfg
from clustered_data import ClusteredScaledDataGenerator

logging.basicConfig(level=logging.DEBUG)
ssl._create_default_https_context = ssl._create_unverified_context


class SGDRegressorClient(fl.client.NumPyClient):
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        train_sample: int = None,
        name=cfg.CLIENT_ID,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.train_sample = train_sample
        self.name = name

        self.edge_model = SGDRegressor()
        self.federated_model = SGDRegressor()

        self.n_features = len(self.X_train.columns)

        self._set_model_zero_coefs(self.edge_model, self.n_features)
        self._set_model_zero_coefs(self.federated_model, self.n_features)

        logging.debug(
            f"Initialized {self.name} with "
            f"X_train.shape={self.X_train.shape} X_test.shape={self.X_test.shape}..."
        )

    def _set_model_zero_coefs(self, model: SGDRegressor, n_features: int) -> None:
        """flwr sever calls a random client for initial params, so we have to set them"""
        model.intercept_ = np.array([self.y_train.mean(), ])
        model.coef_ = np.zeros((n_features,))
        model.feature_names_in_ = np.array(self.X_train.columns)

    def _set_model_coefs(self, model: SGDRegressor, params: List[np.ndarray]) -> None:
        """Sets the parameters of a sklean Regression model."""
        model.intercept_ = params[0]
        model.coef_ = params[1]

    def _get_model_coefs(self, model: SGDRegressor) -> List[np.ndarray]:
        """Returns the paramters of a sklearn LinearRegression model."""
        coefs = [model.intercept_, model.coef_]
        return coefs
    
    def _get_rmse(self, model: SGDRegressor) -> float:
        return mean_squared_error(
            self.y_test.values, model.predict(self.X_test), squared=False
        )

    def get_parameters(self, config):
        params = self._get_model_coefs(self.federated_model)
        logging.debug(f"Client {self.name} sending parameters: {params}")
        return params

    def set_parameters(self, parameters, config):
        logging.debug(
            f"Client {self.name} received parameters {parameters} and {config}"
        )
        self.federated_model.set_params(**config)
        self._set_model_coefs(self.federated_model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)

        if self.train_sample:
            sample_idxs = np.random.choice(len(self.X_train), self.train_sample)
        else:
            sample_idxs = np.arange(len(self.X_train))

        self.edge_model.partial_fit(
            self.X_train.iloc[sample_idxs], self.y_train.iloc[sample_idxs]
        )

        self.federated_model.partial_fit(
            self.X_train.iloc[sample_idxs], self.y_train.iloc[sample_idxs]
        )

        federated_model_coefs = self._get_model_coefs(self.federated_model)
        n_samples = len(sample_idxs)
        fit_metrics = {
            "client_name": self.name,
            "n_samples": n_samples,
        }

        logging.info(f"Client {self.name} fit model with {n_samples} samples.")
        return federated_model_coefs, n_samples, fit_metrics

    

    def evaluate(self, parameters, config): # TODO: check where this parameters come from
        edge_rmse = self._get_rmse(self.edge_model)
        federated_rmse = self._get_rmse(self.federated_model)

        central_model = SGDRegressor()
        central_model.set_params(**config)
        self._set_model_coefs(central_model, parameters)
        central_rmse = self._get_rmse(central_model)

        n_samples = len(self.X_test)

        metrics = {
            "client_name": self.name,
            "n_samples": n_samples,
            "edge_rmse": edge_rmse,
            "federated_rmse": federated_rmse,
            "central_rmse": central_rmse,
        }

        logging.info(
            f"Client {self.name} evaluated rmse: edge={edge_rmse} federated={federated_rmse} central={central_rmse}..."
        )
        return central_rmse, n_samples, metrics


if __name__ == "__main__":
    time.sleep(1)  # wait for server to start
    if cfg.USE_HOUSING_DATA:
        X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    else:
        X, y, coef = make_regression(
            n_samples=20_000,
            n_features=5,
            bias=-2.0,
            n_informative=3,
            noise=1,
            random_state=42,
            coef=True,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = ClusteredScaledDataGenerator(
        pd.DataFrame(X),
        pd.Series(y),
        n_clusters=cfg.N_CLUSTERS,
        test_size=cfg.TEST_SIZE,
        seed=cfg.SEED,
    ).get_random_cluster_train_test_data()
    logging.debug(f"X_train shape: {X_train.shape} y_train shape: {y_train.shape}")
    logging.debug(f"X_test shape: {X_test.shape} y_est shape: {y_test.shape}")

    client = SGDRegressorClient(
        X_train, X_test, y_train, y_test, train_sample=cfg.TRAIN_SAMPLE
    )
    server_address = f"{cfg.SERVER_ADDRESS}:{cfg.SERVER_PORT}"

    while True:
        try:
            fl.client.start_numpy_client(
                server_address=server_address, 
                client=client
            )
        except Exception as e:
            logging.exception(e)
            logging.warning(
                f"Could not connect to server: sleeping for {cfg.RETRY_SLEEP_TIME} seconds..."
            )

        time.sleep(cfg.RETRY_SLEEP_TIME)
