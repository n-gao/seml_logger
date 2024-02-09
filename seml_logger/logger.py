import datetime
import gzip
import json
import logging
import os
import pickle
import shutil
import warnings
from contextlib import contextmanager
from functools import cached_property

import h5py
import numpy as np
import tqdm.rich as tqdm
from seml.json import NumpyEncoder
from tqdm import TqdmExperimentalWarning

from seml_logger.utils import construct_suffix, traverse_tree
from seml_logger.watchers import Watcher
from seml_logger.watchers.aim_watcher import AimWatcher
from seml_logger.watchers.tensorboard_watcher import TensorBoardWatcher


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


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

        self.experiment = experiment
        self.use_tensorboard = use_tensorboard
        self.use_aim = use_aim

        self.log_dir = base_dir
        if experiment is not None:
            self.log_dir = os.path.join(self.log_dir, self.experiment)
        self.run_name = f'{self.name}__{time_str}'
        self.log_dir = os.path.join(self.log_dir, self.run_name)
        self.log_dir = os.path.expanduser(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        self._watchers: list[Watcher] = []
        if self.use_aim:
            self._watchers.append(AimWatcher(
                self.base_dir,
                self.experiment,
                self.name,
                self.log_dir
            ))
        if self.use_tensorboard:
            self._watchers.append(TensorBoardWatcher(
                self.base_dir,
                self.experiment,
                self.name,
                self.log_dir
            ))

        info_dict = {
            'info': self.info_dict,
            'environ': dict(os.environ)
        }
        if config is not None:
            info_dict['hparams'] = config
        for watcher in self.watchers:
            watcher.add_config(info_dict)
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as out:
            json.dump(config, out)
        self._h5py = None
        self.prog = None
    
    @property
    def watchers(self) -> list[Watcher]:
        result = []
        for watcher in self._watchers:
            if not self.use_aim and isinstance(watcher, AimWatcher):
                continue
            if not self.use_tensorboard and isinstance(watcher, TensorBoardWatcher):
                continue
            result.append(watcher)
        return result
    
    @cached_property
    def info_dict(self):
        result = {
            'log_dir': os.path.abspath(self.log_dir),
            'name': self.name
        }
        for watcher in self.watchers:
            result = {**result, **watcher.info()}
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

    def add_distribution(self, path, data, step=None, context=None):
        values = np.array(data).reshape(-1)
        if values.size == 0:
            return
        for watcher in self.watchers:
            watcher.add_distribution(path, values, step, context=context)

    def delete(self):
        self.close()
        path = os.path.abspath(self.log_dir)
        shutil.rmtree(path)

    def close(self):
        for watcher in self.watchers:
            watcher.close()
        if self._h5py is not None:
            self._h5py.close()
    
    def add_tag(self, tag):
        for watcher in self.watchers:
            watcher.add_tag(tag)
    
    def add_text(self, name, value, step=None, context=None):
        for watcher in self.watchers:
            watcher.add_text(name, value, step, context=context)

    def add_distribution_dict(self, tree, path='', step=None, context=None):
        for path, data in traverse_tree(tree, path):
            self.add_distribution(path, data, step=step, context=context)

    def add_scalar(self, name, value, step=None, context=None):
        value = float(value)
        for watcher in self.watchers:
            watcher.add_scalar(name, value, step, context=context)

    def add_scalar_dict(self, tree, path='', step=None, context=None):
        for path, data in traverse_tree(tree, path):
            self.add_scalar(path, float(data), step=step, context=context)
    
    def add_figure(self, name, figure, step=None, context=None):
        for watcher in self.watchers:
            watcher.add_figure(name, figure, step, context=context)
    
    def log_dict(self, tree, path='', step=None, context=None):
        for path, data in traverse_tree(tree, path):
            if isinstance(data, str):
                self.add_text(data, path, step=step, context=context)
                continue
            data = np.array(data)
            if data.size == 1:
                self.add_scalar(path, data, step=step, context=context)
            else:
                self.add_distribution(path, data, step=step, context=context)
    
    def store_result(self, result):
        for watcher in self.watchers:
            watcher.add_result(result)
        self.store_data('result', result, True, True)
    
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
