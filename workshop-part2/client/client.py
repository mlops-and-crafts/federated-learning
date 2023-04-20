import flwr as fl
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class CaliforniaHousingClient(fl.client.NumPyClient):
    def __init__(self):
        self.data = None
        self.target = None
        self.model = LinearRegression()

    def get_weights(self):
        return self.model.coef_, self.model.intercept_

    def fit(self, weights):
        self.model.coef_ = weights[:-1]
        self.model.intercept_ = weights[-1]

    def train(self):
        # Load California housing dataset
        cal_housing = fetch_california_housing()
        self.data = cal_housing.data
        self.target = cal_housing.target

        # Split into train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.target, test_size=0.2, random_state=123
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred_train = self.model.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        print(f"Training MSE: {mse_train:.2f}")

        y_pred_test = self.model.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print(f"Test MSE: {mse_test:.2f}")


if __name__ == "__main__":
    # Start Flower client
    client = CaliforniaHousingClient()
    fl.client.start_numpy_client(server_address="federated-learning-server-1:8080", client=client)
