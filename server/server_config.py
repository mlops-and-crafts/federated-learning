import os

SLEEP_TIME_BETWEEN_ROUNDS = os.environ.get("FEDERATED_SLEEP_TIME", 10)
MIN_CLIENTS = os.environ.get("FEDERATED_MIN_CLIENTS", 2)
NUM_ROUNDS = os.environ.get("FEDERATED_NUM_ROUNDS", 200)
ROUND_TIMEOUT = os.environ.get("FEDERATED_ROUND_TIMEOUT", 5)
PORT = os.environ.get("FEDERATED_PORT", 8080)