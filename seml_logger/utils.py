import json
import os
import pickle
from typing import Mapping
import numpy as np

from seml.utils import flatten

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX.summary import hparams


def traverse_tree(tree, path='', delimiter='/'):
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            new_path = path+delimiter+k if len(path) > 0 else k
            for r in traverse_tree(v, new_path, delimiter):
                yield r
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            k = str(i)
            new_path = path+delimiter+k if len(path) > 0 else k
            for r in traverse_tree(v, new_path, delimiter):
                yield r
    elif tree is None:
        return
    else:
        yield path, tree


def construct_suffix(config, naming, delimiter='_'):
    if naming is not None:
        flat_config = flatten(config)

        def to_name(x):
            if x not in flat_config:
                return 'False'
            val = flat_config[x]
            if isinstance(val, (str, bool, int, float)):
                return str(val)
            else:
                return str(val is not None)
        suffix = delimiter + delimiter.join([to_name(n) for n in naming])
    else:
        suffix = ''
    return suffix


def add_hparams_inplace(writer, hparam_dict, metric_dict, global_step=None):
    if not isinstance(hparam_dict, dict) or not isinstance(metric_dict, dict):
        raise TypeError()
    metric = {
        f'result/{k}': v
        for k, v in metric.items()
    }

    exp, ssi, sei = hparams(hparam_dict, metric_dict)

    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metric.items():
        writer.add_scalar(k, v, global_step)
    writer._get_comet_logger().log_parameters(hparam_dict, step=global_step)


def get_config(path: str):
    path = os.path.expandvars(os.path.expanduser(path))
    with open(os.path.join(path, 'config.json')) as inp:
        return json.load(inp)


def get_result(path: str):
    path = os.path.expandvars(os.path.expanduser(path))
    with open(os.path.join(path, 'result.pickle'), 'rb') as inp:
        return pickle.load(inp)


def get_event_accumulator(path: str, size_guidance: dict = None):
    # By default load only scalars
    sizes = {
        'scalars': 1_000_000,
        'histograms': 0,
        'compressedHistograms': 0,
        'images': 0,
        'audio': 0,
        'tensors': 0
    }
    if size_guidance is not None:
        sizes.update(size_guidance)
    tb_runs = [os.path.join(path, f) for f in os.listdir(path) if '.tfevents' in f]
    tb = tb_runs[np.argmax([os.path.getsize(p) for p in tb_runs])]
    accumulator = EventAccumulator(tb, sizes)
    accumulator.Reload()
    return accumulator
