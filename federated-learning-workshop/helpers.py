from typing import Tuple, Union, List, Dict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

import cfg


def get_test_rmse_from_parameters(
    parameters, config, X_test: pd.DataFrame, y_test: pd.Series
):
    model = SGDRegressor()
    model.set_params(**config)
    model.intercept_ = parameters[0]
    model.coef_ = parameters[1]
    model.feature_names_in_ = np.array(X_test.columns.tolist())

    rmse = mean_squared_error(y_test.values, model.predict(X_test), squared=False)
    return rmse


class ClusteredScaledDataGenerator:
    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_clusters: int = cfg.N_CLUSTERS,
        cluster_cols: List[str] = None,
        seed: int = cfg.SEED,
        test_size: float = cfg.TEST_SIZE,
        strategy: str = "kmeans",
    ):
        self.n_clusters = n_clusters
        self.cluster_cols = cluster_cols
        self.seed = seed
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            pd.DataFrame(X), pd.Series(y), test_size=test_size, random_state=seed
        )
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
        )
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.X_test.columns,
        )

        if strategy == "kmeans":
            self.cluster_ids = self._kmeans_cluster_ids()
        elif strategy == "random":
            self.cluster_ids = self._random_cluster_ids()
        else:
            raise ValueError(
                f"strategy {strategy} not supported. Use either 'kmeans' or 'random'"
            )

    def _random_cluster_ids(self) -> np.ndarray:
        """assign each data point in X_train a random cluster id between 0 and self.n_clusters"""
        return np.random.choice(self.n_clusters, size=len(self.X_train))

    def _kmeans_cluster_ids(self) -> np.ndarray:
        if self.cluster_cols is not None and isinstance(self.X_train, pd.DataFrame):
            X = self.X_train[self.cluster_cols]
        else:
            X = self.X_train
        return KMeans(
            n_clusters=self.n_clusters, random_state=self.seed, n_init=10
        ).fit_predict(X)

    def get_train_test_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_cluster_train_test_data(
        self, cluster_id: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if cluster_id > self.n_clusters:
            raise ValueError(
                f"cluster_id {cluster_id} is larger than number of clusters {self.n_clusters}"
            )
        X = self.X_train[self.cluster_ids == cluster_id]
        y = self.y_train[self.cluster_ids == cluster_id]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed
        )
        return X_train, X_test, y_train, y_test

    def get_random_cluster_train_test_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Return X, y with where self.cluster_ids equals a random cluster_id"""
        np.random.seed()
        X_train = pd.DataFrame()
        while len(X_train) < cfg.TRAIN_SAMPLE:
            cluster_id = np.random.choice(self.n_clusters)
            X_train, X_test, y_train, y_test = self.get_cluster_train_test_data(
                cluster_id
            )
        return X_train, X_test, y_train, y_test


class MetricsJSONstore:
    def __init__(self, metrics_file: str = cfg.METRICS_FILE):
        self.metrics_file = metrics_file
        self.metrics = {"server": [], "clients_fit": [], "clients_evaluate": []}

    def log_server_metrics(
        self, metrics: Dict[str, Union[str, float, np.ndarray]], save=True
    ) -> None:
        self.metrics["server"].append(metrics)
        if save:
            self.save()

    def log_client_fit_metrics(
        self, metrics: Dict[str, Union[str, float, np.ndarray]], save=True
    ) -> None:
        self.metrics["clients_fit"].append(metrics)
        if save:
            self.save()

    def log_client_evaluate_metrics(
        self, metrics: Dict[str, Union[str, float, np.ndarray]], save=True
    ) -> None:
        self.metrics["clients_evaluate"].append(metrics)
        if save:
            self.save()

    def save(self) -> None:
        json.dump(self.metrics, open(self.metrics_file, "w"))

    def load(self) -> Dict:
        if Path(self.metrics_file).exists():
            self.metrics = json.load(open(self.metrics_file, "r"))
        else:
            self.metrics = {"server": [], "clients_fit": [], "clients_evaluate": []}
        return self.metrics
