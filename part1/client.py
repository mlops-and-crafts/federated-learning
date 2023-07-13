import logging
import time
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

import cfg
from helpers import ClusteredScaledDataGenerator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("flwr")


class SGDRegressorClient(fl.client.NumPyClient):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ):
        self.X = X
        self.y = y
        self.federated_model = SGDRegressor()

        # initialize model before fitting because the server will initialize from a random client
        self.federated_model.intercept_ = np.array([0,])
        self.federated_model.coef_ = np.zeros((len(self.X.columns),))
        self.federated_model.feature_names_in_ = np.array(self.X.columns.tolist())

    def get_parameters(self, config) -> List[np.ndarray]:
        """
        method required by flwr.NumPyClient that returns the model coefficients ('parameters')
        to the central server.

        Returns a list of np.ndarrays with the model coefficients.
        """
        return [self.federated_model.intercept_, self.federated_model.coef_]

    def set_parameters(self, parameters, config) -> None:
        """
        method required by flwr.NumPyClient that receives the model coefficients ('parameters') and
        hyperparameters ('config') from the central server, and set them in the client's model.
        """
        self.federated_model.set_params(**config)
        self.federated_model.intercept_ = parameters[0]
        self.federated_model.coef_ = parameters[1]

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """
        method required by flwr.NumPyClient that receives the model coefficients ('parameters')
        and hyperparameters ('config') from the central server, and fits the client's model.

        Returns updated coefficients, number of samples used for fitting, and fit metrics.
        """
        self.set_parameters(parameters, config)

        sample_idxs = np.random.choice(len(self.X), cfg.TRAIN_SAMPLE)

        self.federated_model.partial_fit(
            self.X.iloc[sample_idxs], self.y.iloc[sample_idxs]
        )

        federated_model_coefs = [self.federated_model.intercept_, self.federated_model.coef_]
        n_samples = len(sample_idxs)
        fit_metrics = {
            "client_name": cfg.CLIENT_ID,
            "n_samples": n_samples,
        }
        
        return federated_model_coefs, n_samples, fit_metrics

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        """
        method required by flwr.NumPyClient that receives the model coefficients ('parameters')
        and hyperparameters ('config') from the central server, and evaluates the client's model
        against the local test data.

        Returns loss, number of samples used for evaluation, and evaluation metrics.
        """
        rmse = mean_squared_error(
            self.y.values, self.federated_model.predict(self.X), squared=False
        )

        # can only return metrics singular values such as bool, int, float, etc in metrics dict, not e.g. lists:
        metrics = {
            "client_name": cfg.CLIENT_ID,
            "n_samples": len(self.X),
            "client_rmse": rmse,
        }
        logger.info(f"CLIENT EVAL {cfg.CLIENT_ID} rmse= {rmse:.4f}")
        return rmse, len(self.X), metrics


def get_X_y():
    if cfg.DATASET == 'california_housing':
        X, y = fetch_california_housing(return_X_y=True, as_frame=True)
        cluster_cols=['Latitude', 'Longitude']
    elif cfg.DATASET == 'synthetic':
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
        cluster_cols = ["feature_1", "feature_2"]
    else:
        raise ValueError(f"Unknown dataset {cfg.DATASET}, DATASET must be 'california_housing' or 'synthetic'")

    X_train, _, y_train, _ = ClusteredScaledDataGenerator(
        X,
        y,
        n_clusters=cfg.N_CLUSTERS,
        test_size=cfg.TEST_SIZE,
        seed=cfg.SEED,
        strategy=cfg.CLUSTER_METHOD,
        cluster_cols=cluster_cols,
        
    ).get_random_cluster_train_test_data()
    return X_train, y_train

if __name__ == "__main__":
    X, y = get_X_y()
    time.sleep(1)  # wait for server to start
    
    while True:
        try:
            client = SGDRegressorClient(X, y)
            server_address = f"{cfg.SERVER_ADDRESS}:{cfg.SERVER_PORT}"
            fl.client.start_numpy_client(server_address=server_address, client=client)
        except Exception as e:
            logging.exception(e)
            logging.warning(
                f"Could not connect to server: sleeping for {cfg.RETRY_SLEEP_TIME} seconds..."
            )
        time.sleep(cfg.RETRY_SLEEP_TIME)