# SEML Logger   
This package contains small utility code to enable easy logging to TensorBoardX for experiments managed via [`seml`](https://github.com/TUM-DAML/seml).


## Installation
```
pip install git+https://github.com/n-gao/seml_logger.git
```

## Usage
### Initialization
```python
from sacred import Experiment
from seml_logger import add_logger, Logger

ex = Experiment()

...

def naming_fn(dataset, **_):
    # This function shall return a name for the tensorboard run
    # you may also simply use a timestamp instead but you can also
    # grab any of the main parameters to construct a fitting name.
    return dataset

@ex.automain
@add_logger(naming_fn)
def main(..., dataset='MNIST', logger: Logger):
    ...
```

### Automatic logging
In addition to the manual logs, `seml_logger` also dumps all parameters in a file called `config.json` within the Tensorboard directory.
The config file is also logged in Tensorboard directly via the `Text` functionality.

If the experiment crashes, i.e., the code returns with an exception or an error code != 0, the stacktrace is also logged via Tensorboard for easy access.

Finally, the results of the experiments are stored via `pickle` in `results.pickle` within the logging directory.


### Tensorboard logging
The `Logger` class automatically redirects attributes to `tensorboardX.SummaryWriter`, so you can directly use the logger as
```python
for i in range(10):
    logger.add_scalar(i, global_step=i)
```

Additionally, several utility functions are implemented to log dictionaries of parameters directly.

### File logging
To store parameters or intermediate variables we use `numpy.save` and `numpy.savez`.
You may store arrays in the tensorboard directory via `logger.store_array` or `logger.store_dict`.

### HDF5 logging
To log lots of numerical data, HDF5 presents a nicely accessible way of storing such data.
To create an HDF5 dataset simply call
```python
logger.create_dataset('name', shape=(...), ...)
```
The interface is identical to `h5py.File.create_dataset`. Further, after creating datasets, you may access them directly from the logger via indexing
```python
# works after creating the `name` dataset.
logger['name']
```