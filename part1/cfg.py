import os
import uuid

SERVER_PORT = os.environ.get("SERVER_PORT", 8080)
DATASET = os.environ.get("DATASET", "california_housing") # 'california_housing' or 'synthetic'
CLUSTER_METHOD = os.environ.get("CLUSTER_METHOD", "kmeans") # 'kmeans' or 'iid'
SEED = os.environ.get("CLIENT_SEED", 42)

# server config
SLEEP_TIME_BETWEEN_ROUNDS = os.environ.get("FEDERATED_SLEEP_TIME", 2)
MIN_CLIENTS = os.environ.get("FEDERATED_MIN_CLIENTS", 2)
NUM_ROUNDS = os.environ.get("FEDERATED_NUM_ROUNDS", 2000)
ROUND_TIMEOUT = os.environ.get("FEDERATED_ROUND_TIMEOUT", 5)
METRICS_FILE = os.environ.get("METRICS_FILE", "metrics.json")

# client config
CLIENT_ID = f"client-{os.environ.get('CLIENT_ID', str(uuid.uuid4())[-5:])}"
RETRY_SLEEP_TIME = os.environ.get("CLIENT_RETRY_SLEEP_TIME", 10)
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "0.0.0.0")
TRAIN_SAMPLE = os.environ.get("CLIENT_TRAIN_SAMPLE", 50)
N_CLUSTERS = os.environ.get("CLIENT_N_CLUSTERS", 10)
TEST_SIZE = os.environ.get("CLIENT_TEST_SIZE", 0.2)

# dashboard config
DASHBOARD_PORT = os.environ.get("DASHBOARD_PORT", 8050)