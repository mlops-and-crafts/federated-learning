import os

RETRY_SLEEP_TIME = os.environ.get("CLIENT_RETRY_SLEEP_TIME", 10)
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "localhost")
SERVER_PORT = os.environ.get("SERVER_PORT", "8080")
