from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


class ClusteredDataGenerator:
    def __init__(self, X: np.ndarray, y: np.ndarray, n_clusters: int = 50, seed: int = 42, test_size: float = 0.2):
        self.n_clusters = n_clusters
        self.seed = seed
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(
            X, y, test_size=test_size, random_state=seed)
        self.cluster_ids = self._random_cluster_ids()
        
    def _random_cluster_ids(self) -> np.ndarray:
        """assign each data point in X_train a random cluster id between 0 and self.n_clusters"""
        return np.random.choice(self.n_clusters, size=len(self.X_train))

    def _kmeans_cluster_ids(self) -> np.ndarray:
        raise NotImplementedError
    
    def get_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_cluster_train_test_data(self, cluster_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:    
        X = self.X_train[self.cluster_ids == cluster_id]
        y = self.y_train[self.cluster_ids == cluster_id]
        X_train, X_test, y_train, y_test =  train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed)
        return X_train, X_test, y_train, y_test
    
    def get_random_cluster_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return X, y with where self.cluster_ids equals a random cluster_id"""
        cluster_id = np.random.choice(self.n_clusters)
        return self.get_cluster_train_test_data(cluster_id)