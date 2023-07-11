import json
from typing import Dict, Union

import numpy as np

import cfg

class MetricsJSONstore:
    def __init__(self, metrics_file: str = cfg.METRICS_FILE):
        self.metrics_file = metrics_file
        self.metrics = {"server":[], "clients":[]}

    def log_server_metrics(self, metrics: Dict[str, Union[str, float, np.ndarray]], save=True) -> None:
        self.metrics["server"].append(metrics)
        if save: self.save()

    def log_client_metrics(self, metrics: Dict[str, Union[str, float, np.ndarray]], save=True) -> None:
        self.metrics["clients"].append(metrics)
        if save: self.save()

    def save(self) -> None:
        json.dump(self.metrics, open(self.metrics_file, "w"))

    def load(self) -> Dict:
        self.metrics = json.load(open(self.metrics_file, "r"))
        return self.metrics