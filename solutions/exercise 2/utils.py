import numpy as np


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )


def set_model_params(model, params):
    """Sets the parameters of a sklean Regression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]


def set_initial_params(model):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_features = 8  # Number of features in dataset

    model.coef_ = np.zeros((n_features,))
    if model.fit_intercept:
        model.intercept_ = (0,)
