import flwr as fl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Load the data
data = fetch_california_housing()
X = data["data"]
y = data["target"]

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model

    def get_parameters(self):
        """Return current model parameters."""
        return self.model.coef_, self.model.intercept_

    def fit(self, parameters, config):
        """Update the client model."""
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        return self.train(config)

    def evaluate(self, parameters, config):
        """Evaluate the current model parameters."""
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        return self.test(config)

    def train(self, config):
        """Train the model using the client's data."""
        self.model.fit(X, y)
        return self.get_parameters(), len(X), {}

    def test(self, config):
        """Test the model using the client's data."""
        score = self.model.score(X, y)
        return len(X), score, {}


def main() -> None:
    # Initialize a linear regression model
    model = LinearRegression()

    # Run the Flower server
    # fl.server.start_server("0.0.0.0:8080",
        # FlowerClient(model),
        # config=fl.server.ServerConfig(num_rounds=5),
    # )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 3, "round_duration": 30},
        strategy = FlowerClient(model),
    )

    # fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=5))


if __name__ == "__main__":
    main()




