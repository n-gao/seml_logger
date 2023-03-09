from genericpath import exists
import inspect
import json
import os
import pickle
from typing import Mapping

import numpy as np
import tqdm.auto as tqdm
import seml
from seml.utils import flatten
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from tensorboardX.summary import hparams


def get_experiments(path, return_incomplete=False):
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    experiments = [os.path.join(path, f) for f in os.listdir(path)]
    experiments = [
        p
        for p in experiments
        if  os.path.exists(os.path.join(p, 'config.json'))
        and (os.path.exists(os.path.join(p, 'result.json')) or return_incomplete)
    ]
    result = []
    for experiment in tqdm.tqdm(experiments):
        cfg_file = os.path.join(experiment, 'config.json')
        result_file = os.path.join(experiment, 'result.json')
        result_pkl_file = os.path.join(experiment, 'result.pickle')
        exp = {}
        if os.path.exists(result_pkl_file):
            with open(result_pkl_file, 'rb') as inp:
                exp['result'] = pickle.load(inp)
        else:
            with open(result_file) as inp:
                exp['result'] = json.load(inp)
        with open(cfg_file) as inp:
            exp['config'] = json.load(inp)
        result.append(exp)
    return result


def get_experiments_from_collection(collection, return_incomplete=False):
    cfgs = seml.get_results(collection, fields={'info': True}, states=[])
    experiments = [
        cfg['info']['log_dir']
        for cfg in cfgs
        if 'info' in cfg and 'log_dir' in cfg['info']
    ]
    result = []
    for experiment in tqdm.tqdm(experiments):
        cfg_file = os.path.join(experiment, 'config.json')
        result_file = os.path.join(experiment, 'result.json')
        result_pkl_file = os.path.join(experiment, 'result.pickle')
        exp = {}
        if not os.path.exists(cfg_file):
            continue
        if not os.path.exists(result_file):
            if not return_incomplete:
                continue
        else:
            if os.path.exists(result_pkl_file):
                with open(result_pkl_file, 'rb') as inp:
                    exp['result'] = pickle.load(inp)
            else:
                with open(result_file) as inp:
                    exp['result'] = json.load(inp)
        with open(cfg_file) as inp:
            exp['config'] = json.load(inp)
        result.append(exp)
    return result


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
    path = os.path.expandvars(os.path.expanduser(path))
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


def safe_call(fn, *args, **kwargs):
    params = inspect.signature(fn).parameters
    return fn(*args, **{
        k: v for k, v in kwargs.items()
        if k in params
    })
