import logging
from sacred import Experiment
import numpy as np
import seml

from seml_logger import automain, Logger, add_default_observer_config


ex = Experiment()
seml.setup_logger(ex)

add_default_observer_config(
    ex,
    notify_on_completed=False,
    notify_on_failed=False,
    notify_on_interrupted=False,
    notify_on_started=False
)

def naming_fn(dataset, max_epochs, **_):
    return dataset + '_' + str(max_epochs)


@automain(ex, naming_fn)
def run(dataset: str, hidden_sizes: list, learning_rate: float, max_epochs: int,
        regularization_params: dict, logger: Logger = None):
    # Note that regularization_params contains the corresponding sub-dictionary from the configuration.
    logging.info('Received the following configuration:')
    logging.info(f'Dataset: {dataset}, hidden sizes: {hidden_sizes}, learning_rate: {learning_rate}, '
                 f'max_epochs: {max_epochs}, regularization: {regularization_params}')

    #  do your processing here

    results = {
        'test_acc': 0.5 + 0.3 * np.random.randn(),
        'test_loss': np.random.uniform(0, 10),
        # ...
    }
    # the returned result will be written into the database
    return results
