import datetime
import json
import logging
import pickle
import os
import shutil
from typing import Any
import h5py
import numpy as np
import tensorboardX

from seml_logger.utils import construct_suffix, traverse_tree


class Logger:
    """Logger utility class. This is essentially a wrapper around
    jaxboard with utility functions to log distributions.
    """
    def __init__(
            self,
            name: str,
            naming: list = None,
            config: dict = None,
            folder: str = './logs',
            subfolder: str = None) -> None:
        time_str = datetime.datetime.now().strftime(r'%d-%m-%y_%H:%M:%S:%f')
        self.name = name
        if naming is not None:
            self.name = self.name + construct_suffix(config, naming)
        self.folder_name = folder
        if subfolder is not None:
            self.folder_name = os.path.join(self.folder_name, subfolder)
        self.folder_name = os.path.join(
            self.folder_name, f'{name}__{time_str}')
        self.folder_name = os.path.expanduser(self.folder_name)
        self.writer = tensorboardX.SummaryWriter(self.folder_name)
        if config is not None:
            self.writer.add_text(
                'config', f'```\n{json.dumps(config, indent=2, sort_keys=True)}\n```')
        with open(os.path.join(self.folder_name, 'config.json'), 'w') as out:
            json.dump(config, out)
        self._h5py = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.writer, name)

    def __getitem__(self, index):
        if self._h5py is None:
            raise RuntimeError("No h5py dataset has been created yet!")
        return self.h5py[index]
    
    @property
    def h5py(self):
        if self._h5py is None:
            self._h5py = h5py.File(os.path.join(self.folder_name, 'data.h5py'), 'w')
        return self._h5py

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwargs):
        return self.h5py.create_dataset(name, shape, dtype, data, **kwargs)
    
    def log_distribution(self, data, path, n_bins=20, global_step=None):
        values = np.array(data)
        bins = np.linspace(np.min(values), np.max(values), n_bins)
        self.add_histogram(
            path, bins=bins, values=values, global_step=global_step
        )

    def delete(self):
        self.writer.close()
        path = os.path.abspath(self.folder_name)
        if len(path) < 40:
            logging.info('NOT DELETING BECAUSE THE FOLDER PATH IS ONLY 40 CHARACTERS!')
        else:
            shutil.rmtree(path)
    
    def close(self):
        self.writer.close()
        if self._h5py is not None:
            self._h5py.close()
    
    def log_distribution_dict(self, tree, path='', n_bins=20, global_step=None):
        for path, data in traverse_tree(tree, path):
            self.log_distribution(data, path, n_bins, global_step)

    def log_scalar_dict(self, tree, path='', global_step=None):
        for path, data in traverse_tree(tree, path):
            self.add_scalar(path, data.item(), global_step=global_step)
    
    def store_result(self, result):
        with open(os.path.join(self.folder_name, 'result.pickle'), 'wb') as out:
            pickle.dump(result, out)
    
    def store_array(self, array, filename):
        np.save(os.path.join(self.folder_name, filename), np.array(array))
    
    def store_dict(self, filename, **kwargs):
        np.savez(os.path.join(self.folder_name, filename), **kwargs)
