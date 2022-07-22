import inspect
import traceback
from typing import Iterable

from merge_args import merge_args

from seml_logger.logger import Logger


def add_logger(naming_fn, default_naming=None, default_folder='./logs', subfolder=None):
    def annotation(fn):
        if 'logger' not in inspect.signature(fn).parameters:
            raise ValueError("The decorated function should take `logger` as argument!")
        def func( 
            naming: Iterable[str] = default_naming, 
            folder: str = default_folder,
            subfolder: str = subfolder, 
            db_collection: str = None,
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
            defaults = {k: v for k, v in zip(inspect.getfullargspec(fn).args[::-1], inspect.getfullargspec(fn).defaults[::-1])}
            config = {**defaults, **config, **kwargs}
            if subfolder is None:
                subfolder = db_collection
            logger = Logger(name=naming_fn(**config), naming=naming, config=config, folder=folder, subfolder=subfolder)
            try:
                result = fn(**kwargs, logger=logger)
                # Store results with pickle
                logger.store_result(result)
                logger.close()
                return result
            except Exception as e:
                # Store exception in tensorboard for easier debugging
                logger.add_text('Exception', f'```\n{traceback.format_exc()}\n```')
                logger.close()
                raise e
        result = merge_args(fn)(func)
        # We to do this to ensure that "__main__" is preserved
        result.__module__ = fn.__module__
        return result
    return annotation
