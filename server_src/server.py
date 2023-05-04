import flwr as fl



# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(min_available_clients=2)
    fl.server.start_server(server_address="0.0.0.0:8080",
                           strategy=strategy, config=fl.server.ServerConfig(
                               num_rounds=3))
