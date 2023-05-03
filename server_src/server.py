import flwr as fl
from sklearn.linear_model import LinearRegression
from utils import set_initial_params


# def get_evaluate_fn(model: LinearRegression):
#     """Return an evaluation function for server-side evaluation."""

#     X, y = fetch_california_housing(return_X_y=True)
#     X_test, y_test = X[15000:], y[15000:]

#     def evaluate(
#         server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
#     ):
#         set_model_params(model, parameters)
#         mse = mean_squared_error(y_test, model.predict(X_test))
#         r_squared = model.score(X_test, y_test)
#         return r_squared, {"mse": mse}

#     return evaluate

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LinearRegression()
    set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(min_available_clients=2)
    fl.server.start_server(server_address="0.0.0.0:8080",
                           strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))
