import logging
import time

import flwr as fl
import cfg

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("flwr")

def evaluate_fn(server_round, parameters, config):
    time.sleep(cfg.SLEEP_TIME_BETWEEN_ROUNDS)

if __name__ == "__main__":
    while True:
        try:
            strategy = fl.server.strategy.FedAvg(
                min_available_clients=cfg.MIN_CLIENTS,
                evaluate_fn=evaluate_fn,
            )
            fl.server.start_server(
                server_address=f"0.0.0.0:{cfg.SERVER_PORT}",
                strategy=strategy,
                config=fl.server.ServerConfig(
                    num_rounds=cfg.NUM_ROUNDS,
                    round_timeout=cfg.ROUND_TIMEOUT,
                ),
            )
        except Exception as e:
            logging.exception(e)
