import json
import logging
from typing import Any

import numpy as np
import tensorboardX
from seml.json import NumpyEncoder
from seml_logger.tensorboard_handler import TensorBoardHandler
from seml_logger.utils import traverse_tree

from seml_logger.watchers import Watcher


def context_to_name(name, context):
    if context is None:
        return name
    result = '/'.join(
        f'{path}_{val}'
        for path, val in traverse_tree(context)
    ) + f'/{name}'
    # Several characters are not supported by tensorbard
    result = result.replace(' ', '_')\
        .replace('#', '_')\
        .replace('=', '_')\
        .replace()
    return result


class TensorBoardWatcher(Watcher):
    def __init__(
        self,
        log_dir: str,
        experiment: str,
        name: str,
        experiment_dir: str,
        n_bins: int = 64
    ):
        self.log_dir = log_dir
        self.experiment = experiment
        self.name = name
        self.experiment_dir = experiment_dir
        self.n_bins = n_bins

        self.writer = tensorboardX.SummaryWriter(self.experiment_dir)

        log = logging.getLogger()
        handler = TensorBoardHandler(self.writer)
        if len(log.handlers) > 0:
            handler.setFormatter(log.handlers[-1].formatter)
        log.addHandler(handler)

    def add_scalar(self, name, value, step, context: Any):
        name = context_to_name(name, context)
        self.writer.add_scalar(name, value, step)

    def add_distribution(self, name, data, step, context: Any):
        name = context_to_name(name, context)
        bins = np.linspace(np.min(data), np.max(data), self.n_bins)
        self.writer.add_histogram(name, data, step, bins)

    def add_figure(self, name, figure, step, context: Any):
        name = context_to_name(name, context)
        self.writer.add_figure(name, figure, step)

    def add_text(self, name: str, value: str, step: int, context: Any):
        name = context_to_name(name, context)
        self.writer.add_text(name, value, step)

    def add_result(self, result: dict[str, Any]):
        self.writer.add_text(
            'result',
            f'```\n{json.dumps(result, cls=NumpyEncoder)}\n```'
        )

    def add_config(self, config: dict[str, Any]):
        for key, value in config.items():
            self.writer.add_text(
                key,
                f'```\n{json.dumps(value, cls=NumpyEncoder)}\n```'
            )

    def add_tag(self, tag: str):
        return

    def close(self):
        self.writer.close()
