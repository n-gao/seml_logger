import json
import logging
from typing import Any

import aim
import numpy as np
from seml.utils.json import NumpyEncoder
from seml_logger.utils import ignore_warnings

from seml_logger.watchers.watcher import Watcher


class AimWatcher(Watcher):
    def __init__(self, log_dir: str, experiment: str, name: str, experiment_dir: str):
        self.log_dir = log_dir
        self.experiment = experiment
        self.name = name
        self.run = aim.Run(repo=self.log_dir, experiment=self.experiment)
        self.run.name = name

    def track(self, name: str, trackObj, step: str, context: Any):
        self.run.track(trackObj, name, step, context=context)

    def add_scalar(self, name: str, value: float, step: int, context: Any):
        self.track(name, value, step, context=context)

    def add_distribution(self, name: str, data: np.ndarray, step: int, context: Any):
        self.track(name, aim.Distribution(data), step, context=context)

    def add_figure(self, name: str, figure, step: int, context: Any):
        try:
            with ignore_warnings():
                value = aim.Figure(figure)
        except:
            value = aim.Image(figure)
        self.track(name, value, step, context=context)

    def add_text(self, name: str, value: str, step: int, context: Any):
        self.track(name, aim.Text(value), step, context=context)

    def add_result(self, result: dict[str, Any]):
        # We convert to JSON and back to avoid storing non-native variables
        try:
            # We convert to JSON and back to avoid storing non-native variables
            self.run["result"] = json.loads(json.dumps(result, cls=NumpyEncoder))
        except TypeError as e:
            logging.warn(str(e))

    def add_config(self, config: dict[str, Any]):
        for key, value in config.items():
            self.run[key] = value

    def add_tag(self, tag: str):
        return self.run.add_tag(tag)

    def info(self):
        return {"aim": {"hash": self.run.hash}}

    def close(self):
        self.run.close()
