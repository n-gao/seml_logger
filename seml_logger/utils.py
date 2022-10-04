import json
import os
import pickle

from seml.utils import flatten

from tensorboardX.summary import hparams


def traverse_tree(tree, path='', delimiter='/'):
    if isinstance(tree, dict):
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
