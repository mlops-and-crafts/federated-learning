import flwr as fl

strategy = fl.server.strategy.FedAvg(
    min_available_clients=3,  # Minimum number of clients that need to be connected to the server before a training round can start
)

if __name__ == "__main__":
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=5),strategy = strategy)