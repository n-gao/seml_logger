import inspect
from io import StringIO
import logging
import traceback
from typing import Iterable

from merge_args import merge_args
from sacred import Experiment
from sacred.observers import MongoObserver

from seml_logger.logger import Logger


def _get_mongodb_observer(experiment: Experiment):
    for obs in experiment.observers:
        if isinstance(obs, MongoObserver):
            return obs
    return None


def add_logger(experiment: Experiment, naming_fn, default_naming=None, default_folder='./logs', subfolder=None):
    def annotation(fn):
        if 'logger' not in inspect.signature(fn).parameters:
            raise ValueError(
                "The decorated function should take `logger` as argument!")

        def func(
                naming: Iterable[str] = default_naming,
                folder: str = default_folder,
                subfolder: str = subfolder,
                db_collection: str = None,
                print_progress: bool = False,
                **kwargs):
            # To get the whole config we have to remove the additional params we got from the `add_logger` function
            config = locals()
            for k in inspect.signature(add_logger).parameters.keys():
                if k in config:
                    del config[k]
            # We don't want to log the decorated function
            del config['fn']
            # Also we do not want to replicate the overwritten parameters
            del config['kwargs']
            # Also we have to add the default values of the original function
            defaults = {k: v for k, v in zip(inspect.getfullargspec(
                fn).args[::-1], inspect.getfullargspec(fn).defaults[::-1])}
            config = {**defaults, **config, **kwargs}

            # Initialize logger
            if subfolder is None:
                subfolder = db_collection
            logger = Logger(name=naming_fn(**config), naming=naming,
                            config=config, folder=folder, subfolder=subfolder, print_progress=print_progress)
            logging.info(f'TensorBoard: {logger.folder_name}')

            # Capture all logging
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            logging.getLogger().addHandler(handler)

            # Add logdir to MongoDB
            observer = _get_mongodb_observer(experiment)
            if observer is not None:
                if observer.run_entry is not None:
                    observer.run_entry['log_dir'] = logger.folder_name
                    observer.save()
            
            # Actually run experiment
            try:
                result = fn(**kwargs, logger=logger)
                # Store results with pickle
                logger.store_result(result)
                logger.add_text('Log', f'```\n{stream.getvalue()}\n```')
                logger.close()
                return result
            except Exception as e:
                # Store exception in tensorboard for easier debugging
                logger.add_text(
                    'Exception', f'```\n{traceback.format_exc()}\n```')
                logger.add_text('Log', f'```\n{stream.getvalue()}\n```')
                logger.close()
                raise e
        result = merge_args(fn)(func)
        # We to do this to ensure that "__main__" is preserved
        result.__module__ = fn.__module__
        return result
    return annotation


def automain(experiment: Experiment, naming_fn, default_naming=None, default_folder='./logs', subfolder=None):
    annotate_fn = add_logger(experiment, naming_fn,
                             default_naming, default_folder, subfolder)

    def annotate(fn):
        return experiment.automain(annotate_fn(fn))
    return annotate


def main(experiment: Experiment, naming_fn, default_naming=None, default_folder='./logs', subfolder=None):
    annotate_fn = add_logger(experiment, naming_fn,
                             default_naming, default_folder, subfolder)

    def annotate(fn):
        return experiment.main(annotate_fn(fn))
    return annotate
