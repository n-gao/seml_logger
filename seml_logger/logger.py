from contextlib import contextmanager
import datetime
import functools
import gzip
import json
import logging
import os
import pickle
import shutil
import time

import aim
import h5py
import numpy as np
import tensorboardX
import tqdm.auto as tqdm
from numpy.distutils.misc_util import is_sequence
from seml.utils import flatten
from seml.json import NumpyEncoder

from seml_logger.utils import add_hparams_inplace, construct_suffix, traverse_tree


def _try_create_aim_run(repo, experiment, ntries=5):
    try:
        return aim.Run(repo=repo, experiment=experiment)
    except Exception as e:
        if ntries > 0:
            time.sleep(1.0)
            return _try_create_aim_run(repo, experiment, ntries - 1)
        raise e


class Logger:
    """Logger utility class.
    This is a wrapper around aim.Run and tensorboardX with utility functions to log distributions.
    """
    def __init__(
            self,
            name: str,
            naming: list = None,
            config: dict = None,
            base_dir: str = './logs',
            aim_repo: str = None,
            experiment: str = None,
            print_progress: bool = False,
            use_tensorboard: bool = True,
            use_aim: bool = True) -> None:
        time_str = datetime.datetime.now().strftime(r'%d-%m-%y_%H:%M:%S:%f')
        self.name = name
        self.print_progress = print_progress
        self.config = config
        if naming is not None:
            self.name = self.name + construct_suffix(config, naming)
        self.base_dir = base_dir
        self.aim_repo = aim_repo if aim_repo is not None else self.base_dir
        self.log_dir = base_dir
        self.experiment = experiment
        self.use_tensorboard = use_tensorboard
        self.use_aim = use_aim

        if experiment is not None:
            self.log_dir = os.path.join(self.log_dir, self.experiment)
        self.run_name = f'{self.name}__{time_str}'
        self.log_dir = os.path.join(self.log_dir, self.run_name)
        self.log_dir = os.path.expanduser(self.log_dir)

        if use_tensorboard:
            self.tb_writer = tensorboardX.SummaryWriter(self.log_dir)
        else:
            os.makedirs(self.log_dir, exist_ok=True)
        
        if use_aim:
            self.aim_run = _try_create_aim_run(self.aim_repo, self.experiment)
            self.aim_run.name = self.name

        if config is not None:
            if self.use_tensorboard:
                self.tb_writer.add_text(
                    'config', f'```\n{json.dumps(config, indent=2, sort_keys=True)}\n```')
            if self.use_aim:
                self.aim_run['hparams'] = config
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as out:
            json.dump(config, out)
        self._h5py = None
        self.prog = None
    
    @property
    def info_dict(self):
        result = {
            'log_dir': self.log_dir,
            'name': self.name
        }
        if self.use_aim:
            result['aim_hash'] = self.aim_run.hash
        return result

    def __getitem__(self, index):
        if self._h5py is None:
            raise RuntimeError("No h5py dataset has been created yet!")
        return self.h5py[index]

    @contextmanager
    def without_aim(self):
        prev_value = self.use_aim
        try:
            self.use_aim = False
            yield self
        finally:
            self.use_aim = prev_value

    @contextmanager
    def without_tensorboard(self):
        prev_value = self.use_tensorboard
        try:
            self.use_tensorboard = False
            yield self
        finally:
            self.use_tensorboard = prev_value

    @property
    def h5py(self):
        if self._h5py is None:
            self._h5py = h5py.File(os.path.join(
                self.log_dir, 'data.h5py'), 'w')
        return self._h5py

    def tqdm(self, iterator, *args, **kwargs):
        self.prog = tqdm.tqdm(iterator, *args, **kwargs,
                              disable=not self.print_progress)
        return self.prog

    def set_postfix(self, postfix):
        if self.prog is None:
            raise RuntimeError()
        self.prog.set_postfix(postfix)

    def create_dataset(self, name, shape=None, dtype=None, data=None, compression='gzip', **kwargs):
        return self.h5py.create_dataset(name, shape, dtype, data, compression=compression, **kwargs)

    def add_distribution(self, data, path, n_bins=64, step=None, context=None):
        values = np.array(data).reshape(-1)
        if values.size == 0:
            return
        if self.use_aim:
            self.aim_run.track(aim.Distribution(values), name=path, step=step, context=context)
        if self.use_tensorboard:
            if context is not None and 'subset' in context:
                path = context['subset'] + '/' + path
            bins = np.linspace(np.min(values), np.max(values), n_bins)
            self.tb_writer.add_histogram(
                path, bins=bins, values=values, global_step=step
            )

    def delete(self):
        self.close()
        path = os.path.abspath(self.log_dir)
        if len(path) < 40:
            logging.info(
                'NOT DELETING BECAUSE THE FOLDER PATH IS ONLY 40 CHARACTERS!')
        else:
            shutil.rmtree(path)

    def close(self):
        if self.use_tensorboard:
            self.tb_writer.close()
        if self.use_aim:
            self.aim_run.close()
        if self._h5py is not None:
            self._h5py.close()
    
    def add_tag(self, tag):
        if self.use_aim:
            self.aim_run.add_tag(tag)
    
    def add_text(self, name, value, step=None, context=None):
        if self.use_aim:
            self.aim_run.track(aim.Text(value), name, step=step, context=context)
        if self.use_tensorboard:
            if context is not None and 'subset' in context:
                name = context['subset'] + '/' + name
            self.tb_writer.add_text(name, value, global_step=step)

    def add_distribution_dict(self, tree, path='', n_bins=64, step=None, context=None):
        for path, data in traverse_tree(tree, path):
            self.add_distribution(data, path, n_bins=n_bins, step=step, context=context)

    def add_scalar(self, name, value, step=None, context=None):
        if self.use_aim:
            self.aim_run.track(value, name, step=step, context=context)
        if self.use_tensorboard:
            if context is not None and 'subset' in context:
                name = context['subset'] + '/' + name
            self.tb_writer.add_scalar(name, value, global_step=step)

    def add_scalar_dict(self, tree, path='', step=None, context=None):
        for path, data in traverse_tree(tree, path):
            self.add_scalar(path, data.item(), step=step, context=context)
    
    def add_figure(self, name, figure, step=None, context=None):
        if self.use_aim:
            self.aim_run.track(aim.Image(figure), name, step=step, context=context)
        if self.use_tensorboard:
            if context is not None and 'subset' in context:
                name = context['subset'] + '/' + name
            self.tb_writer.add_figure(name, figure, global_step=step)
    
    def log_dict(self, tree, path='', step=None, context=None):
        for path, data in traverse_tree(tree, path):
            if isinstance(data, str):
                self.add_text(data, path, step=step, context=context)
                continue
            data = np.array(data)
            if data.size == 1:
                self.add_scalar(path, data, step=step, context=context)
            else:
                self.add_distribution(data, path, step=step, context=context)
    
    def store_result(self, result):
        with self.without_aim():
            self.log_dict(result, 'result', context={'subset': 'result'})
        self.store_data('result', result, True, True)
        if isinstance(result, dict):
            try:
                config = flatten(self.config)
                flat_result = flatten(result)
                for k in flat_result:
                    if is_sequence(flat_result[k]):
                        flat_result[k] = np.mean(flat_result[k]).item()
                with self.without_aim():
                    add_hparams_inplace(self, config, flat_result)
            except:
                pass
        if self.use_aim:
            try:
                # We convert to JSON and back to avoid storing non-native variables
                self.aim_run['result'] = json.loads(json.dumps(result, cls=NumpyEncoder))
            except TypeError as e:
                logging.warn(str(e))
    
    def store_data(self, filename, data, use_json=False, use_pickle=True, use_gzip=False):
        open_fn = gzip.open if use_gzip else open
        suffix = '.gz' if use_gzip else ''
        if use_json:
            try:
                mode = 'wb' if use_gzip else 'w'
                with open_fn(os.path.join(self.log_dir, f'{filename}.json{suffix}'), mode) as out:
                    json.dump(data, out, cls=NumpyEncoder)
            except TypeError as e:
                logging.warn(str(e))
        if use_pickle:
            with open_fn(os.path.join(self.log_dir, f'{filename}.pickle{suffix}'), 'wb') as out:
                pickle.dump(data, out)

    def store_array(self, filename, array):
        np.save(os.path.join(self.log_dir, filename), np.array(array))

    def store_dict(self, filename, **kwargs):
        np.savez_compressed(os.path.join(self.log_dir, filename), **kwargs)
    
    def store_blob(self, filename, blob):
        path = os.path.join(self.log_dir, filename)
        with open(path, 'wb') as out:
            out.write(blob)
