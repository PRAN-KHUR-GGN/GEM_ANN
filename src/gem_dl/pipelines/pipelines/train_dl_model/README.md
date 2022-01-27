# Deep Learning Pipeline
The deep learning pipeline provides a pipeline for tuning and training a neural network predictive models.

## Overview
In certain situations, neural network-based deep learning models are the most appropriate fit for predictive modeling of 
time-series datasets, particularly in cases where there is a large volume of data, users have access to distributed compute,
and there is a need to improve predictive power and extrapolation capabilities. One of the most critical steps in building 
a deep learning model is in finding appropriate and well performing hyper-parameters. This pipeline provides nodes that 
help a data scientist easily parameterize their hyper-parameter tuning run, fit the resulting model, and report on it's 
performance.

## Parameters
The parameters file for the deep learning can be broken down into 2 sections, the first defines the parameters and setting for building and training a *single* model, and the second sets up the hyper-parameter tuning procedure.

### Block 1: The Model
The choice of model architecture has critical implications on what the format of our input data must be. Concretely, feedforward networks (e.g. Multi-Layer Perceptrons, or MLPs) only require flat data, while recurrent and convolutional networks require data formatted as a tensor of at least 3-dimensions.

Within the paramaters file, users specify the high-level architecture, either leveraging the Sequential or Functional APIs of `tf.keras`. If hyper-parameter tuning is used to chose a value of one of the hyper-parameters specified in this model block, then then final tuned model will use the value chosen, not the value specified.

A typical modeling parameters block looks like:

```yaml
train_deeplearning:
    target: 'model_target' # name of column where the target_tag is indicated in td
    features: 'model_feature' # name of column where features for this model are stored in tag dictionary

    split:
        type: frac
        train_split_fract: 0.8 # required only when `type: frac`
        datetime_val: !!timestamp '2020-04-26 03:59:59' # required only when `type: date`
        datetime_col: "timestamp"
    transformers:
        - class: sklearn.preprocessing.StandardScaler
          kwargs: {}
          name: standard_scaler
    build_fn: core_pipelines.pipelines.train_dl_model.nodes.create_sequential_model # path to tf model building function.
    estimator: # path to Keras wrapper class (usually either `KerasRegressor` or `KerasClassifier`).
        class: tensorflow.keras.wrappers.scikit_learn.KerasRegressor
    fit:
        epochs: 50 # How many epochs to train our model once we've found the best hyper-parameters.
        callbacks: # Specification of callbacks to be used for tracking training progress.
            tensorboard:
                class: 'tensorflow.keras.callbacks.TensorBoard'
                kwargs:
                    log_dir: "<path/to/logs>"
            es:
                class: 'tensorflow.keras.callbacks.EarlyStopping'
                kwargs:
                    patience: 100
                    monitor: 'val_loss'
            ray: # ray[tune] keras integration callback. Required for hyperparameter tunning
                class: ray.tune.integration.keras.TuneReportCallback
                kwargs:
                    metrics: "val_loss"
                    "on": "epoch_end"
    metrics: # Strings to be passed to keras model compilation. These will used to track progress of fitting
        - 'mean_absolute_error'
        - 'mean_absolute_percentage_error'
    architecture:
        # Here we'll lay out a 3 layer MLP architecture
        layer_01:
            n_block: 1 # number of times to repeat this layer
            class: "tensorflow.keras.layers.Dense" # type of layer (e.g. dense, convolutional etc).
            kwargs:
                units: 20 # size of layer (number of units/activations)
                activation: "relu" # activation function
        layer_02:
            n_block: 1
            class: "tensorflow.keras.layers.Dense"
            kwargs:
                units: 10
                activation: "relu"
        layer_03:
            n_block: 1
            class: "tensorflow.keras.layers.Dense"
            kwargs:
                units: 1
                activation: "linear"

    optimizer: # These are the parameters of the optimization algorithm to be used for fitting the model.
        class: 'tensorflow.keras.optimizers.Adam' # optimization class specification (inside tensorflow)
        kwargs: # keyword args to optimization constructor, change depending on choice of opt. algorithm.
            learning_rate: 0.005 # default adam lr = 0.001
    loss: # Specification of loss, which will determine how well our predictions are doing.
        MeanAbsoluteError:
            class: "tensorflow.keras.losses.MeanAbsoluteError"
            kwargs: {} # Most losses don't accept extra arguments

```

#### Modeling Using Recurrent Neural Network Architectures
There's also the ability to include `data reformatters` if the chosen architecture dictates. These will transform a flat file to a `3+`-dimensional tensor, as needed.

A very useful utility for transforming data into the appropriate format can be found at
`tf.keras.preprocessing.sequence.TimeSeriesGenerator`. This accepts data and target values,
and transforms the flat file into a tensor taking into account arguments including stride,
length (of time-window) etc, and will be needed to be added to the parameters. This might result in the top of our modeling block within parameters looking like:

```yaml
train_deeplearning:
    transformers:
        - class: sklearn.preprocessing.StandardScaler
          kwargs: {}
          name: standard_scaler
# The reshaper below is for leveraging *time series models explicitly. We'll always be assuming the "Model" that is the sklearn pipeline will be accepted 2-D data.
   data_reformatter:
       class: tensorflow.keras.preprocessing.sequence.TimeseriesGenerator
       kwargs:
           length: 128 # This is the window length to lookback on
           shuffle: False

```

When using reformatters to apply recurrent architectures, a few assumptions must be met:

1) The first/primary axis (axis 0) represents the `time` variable, and rows are present in ascending order.
2) There are no missing data entries. Each time interval of interest is present in the data.

Generally speaking, the input data for a RNN-type model will be at a higher granularity than that which is used for other model architectures/algorithms, though the level of granularity will be user and context specific. Importantly, the length of time window to use will be context specific,but can be tuned during training by modifying the `length` parameter of the data formatter. It is *critical* that this value be set properly for later model and counterfactual evaluation.

Many downstream nodes, are explicitly built around the assumption that a call to `model.predict(x)`, will have the same size along axis-0 as `x`, but this assumption no longer holds when a data-formatter is used within an instance of `KerasEstimator`. The shape of a call to `model.predict` will then  depend upon parameters including the `window_length` and `stride`. Here the `stride` determines how separated subsequent predictions should be (e.g. 1 row, 2 rows etc).

#### On target variable definition
When using the LSTM, the model will be leveraging a higher-granularity dataset, with individual timesteps differing potentially on the 5-10 minute level. The model prediction will be taking not an individual time point, but a time-series of input features, and handles the necessary reformatting of data with the data_reformatter attribute, as described above. This construction assumes that the target variable is already lagged appropriately (if needed). That is, if we are making a prediction at 12:00pm of throughput, when training the target variable for that timestep is throughput at 1600 (if using a look-ahead window of 4hrs), and not throughput at 1200. The data-reformatter does *not* handle a custom-prediction lookahead.

### Block 2: Hyperparameter tuning
The deep learning pipeline leverages the `tune` library within the software package [ray](https://docs.ray.io/) in for hyper-parameter tuning. This achieves two goals, it allows for straightforward expansion and utilization of large, distributed computation, and provides a unified-interface to a number of bespoke hyperparameter tuning algorithms.

To perform hyper-parameter search using `ray.tune`, one must make two design choices- that of `scheduler` and that of `search` algorithms. Search algorithms suggest new candidate configurations/parameter sets to evaluate, and schedulers track performance of fitting during individual experiments, potentially stopping trials early which are not likely to perform well.

A typical tuning params block looks like:

```yaml
    tune_params:
        metric: val_loss # the metric to optimize when performing hyper-parameter tuning
        mode: min # [min/max] -> whether configurations should max/minimize the metric.
        config: # Definition of the search space of hyper-parameters
          layer_01_units:
            class: ray.tune.randint
            kwargs:
                lower: 16
                upper: 32
          learning_rate:
            class: ray.tune.uniform
            kwargs:
                lower: !!float 0.0001
                upper: !!float 0.005
        scheduler:
            class: ray.tune.schedulers.AsyncHyperBandScheduler
            kwargs:
                time_attr: training_iteration
                max_t: 40
                grace_period: 20
        # search_alg: # Users would specify class and kwargs of search_alg, if using. See below for details
        #    class:
        #    kwargs:
        fail_fast: True
        local_dir: path/to/logs # location for logging tuning results from `ray`
        num_samples: 8
        checkpoint_freq: 2 # How frequently (in epochs) to checkpoint the model
        verbose: 0 # level of messages to print
        stop:
            training_iteration: 10
        resources_per_trial: # how to allocate trials to resources
            cpu: 1
            gpu: 0
```

#### `config_mapper`

To parse the hyperparameters returned by `ray.tune` during optimization, we need a function
that is able to translate them into a `train_deeplearning` configuration like the one in Block 1.
For the default hyperparameter search above, we use as
default the following `config_mapper`:


```python
def map_config_params(params, config):
    """
    Map config settings to params dictionary.

    This updated parameter set allows us to potentially
    search over a complete parameter set of neural net architecture parameters,
    and optimizer parameters.

    Args:
        params: Initial parameter dictionary defining how to construct our DL model.
        config: The worker's configuration parameters, which we want to test
        performance of. This is a dictionary created from the call to `ray.tune.run`

    Returns: updated params dictionary.

    """
    params["optimizer"]["kwargs"]["learning_rate"] = config.get("learning_rate", 0.005)
    params["architecture"]["layer_01"]["kwargs"]["units"] = config.get(
        "layer_01_units", 10
    )
    return params
```

You are free to use your own `config_mapper` by passsing it to 
`create_pipeline`

`train_dl_model.create_pipeline(my_config_mapper)`

### Block 3: Reporting
Here we'll specify which parameters for the model performance report (the last node in the pipeline).
These are very similar to the block in the standard `train_model` pipeline parameters. 

```yaml
    report:
        output_path: ${base_dir}${folders.mill_demo}/train/08_reporting
        report_name: performance_report
        timestamp: True
        title: "Model Performance Report"
        author: "OptimusAI"
        subject: "OptimusAI Performance Report"

```
#### shap
`shap` is not installed by default, to leverage the shap reporting nodes available in `core_pipelines.pipelines.utils.shap_plots`, you can install shap either via `pip install shap` or via `pip install core_pipelines[shap]`. You can then add the `generate_shap_figures` to your pipeline - it will generate a dictionary of shap plots for your model. Additional functions for generating individual level shap plots are also available in the `shap_reports` module.

You need to pass extra params to this function, to do so, please add a block like the below to your params.

```yaml
shap:
    explainer: shap.DeepExplainer
    kwargs: {}
```

#### Custom search algorithms
The `ray.tune` package provides an interface to many alternative hyper-parameter
search algorithms, which can be used to suggest values of hyper-parameters from a user-specified grid. By default this pipeline *does not* leverage any package/approach which requires the installation of
additional dependencies.

Using an alternative search method requires specifying appropriate search_algorithm parameters.

To do this, add other associated parameters to the parameters file, specifically the following block within the `tune_params` block:
```yaml
class: ray.tune.suggest.hyperopt.HyperOptSearch
kwargs:
    metric: loss
    mode: min
```


Please refer to the `ray` [docs](https://docs.ray.io/) for additional details on the available search algorithms and trial schedulers.

## Customizing the tensorflow model
To define a custom model, and retain the ability to hyper-parameter tune over arbitrary
model choices, the user will need to provide two functions, along with suitable changes to the 
`params` block (discussed above).

### The `build_fn`
The default `build_fn` builds a sequential model from the params and architecture 
supplied by the user. If a more complex architecture and model build is needed, users need to supply a 
function that takes a configuration file and returns a compiled keras model. This will enable users to 
specify a build function that leverages the sequential OR functional keras api.

The pattern of using these model build functions can also be seen in other frameworks, like [keras-tuner](https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html).

### The `config_mapper`
Users also need to supply a config_mapper when using the included tuning
pipeline. This can be passed as an argument to the
`core_pipelines.pipelines.train_dl_model.create_pipeline` function that
instantiates the `train_dl_model` modular pipeline. For example, at `pipeline_registry.py`

```python

from typing import Dict

from kedro.pipeline import Pipeline
from core_pipelines.pipelines.train_dl_model import create_pipeline
from .utils import my_config_mapper


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "__default__": create_pipeline(config_mapper=my_config_mapper)
    }
```

`config_mapper` does one job -
it maps an example configuration parameter to the appropriate parameters block.

If no `config_mapper` is passed, the following is used as default:

```python
def map_config_params(params, config):
    """
    Map config settings to params dictionary.

    This updated parameter set allows us to potentially
    search over a complete parameter set of neural net architecture parameters,
    and optimizer parameters.

    Args:
        params: Initial parameter dictionary defining how to construct our DL model.
        config: The worker's configuration parameters, which we want to test
        performance of. This is a dictionary created from the call to `ray.tune.run`

    Returns: updated params dictionary.

    """
    if "learning_rate" in config:
        params["optimizer"]["kwargs"]["learning_rate"] = config.get(
            "learning_rate", 0.005
        )
    if "layer_01_units" in config:
        params["architecture"]["layer_01"]["kwargs"]["units"] = config.get(
            "layer_01_units", 10
        )
    return params
```


which assumes we are only doing hyperparameter search over the learning rates and size
of the first layer.

### Custom DL Arch examples

#### MLP

MLPs are some of the most effective model types for optimus projects. Here we'll
see how to do hyperparameter search over a large space of architectures.


**Hyperparameter Search Space**

The following search space searches for architectures with 1 or 2 hidden layers,
different dropout rates on the hidden layers, different number of units at each
hidden layer, different activation functions for the hidden layers and final
activation, different learning rates, and different optimizers.

```yaml
    tune_params:
        config:
            two_hidden_layers:
                class: ray.tune.choice
                kwargs:
                    categories: [True, False]
            dropout1:
                class: ray.tune.uniform
                kwargs:
                    lower: !!float 0.01
                    upper: !!float 0.6
            dropout2:
                class: ray.tune.uniform
                kwargs:
                lower: !!float 0.01
                upper: !!float 0.6
            units1:
                class: ray.tune.uniform
                kwargs:
                lower: 15
                upper: 40
            units2:
                class: ray.tune.uniform
                kwargs:
                lower: 5
                upper: 30
            internal_activation:
                class: ray.tune.choice
                kwargs:
                categories: ["swish", "softplus", "selu", "relu", "sigmoid"]
            final_activation:
                class: ray.tune.choice
                kwargs:
                categories: ["swish", "softplus", "selu"]
            initial_learning_rate:
                class: ray.tune.loguniform
                kwargs:
                    lower: !!float 0.06
                    upper: !!float 0.005
            optimizer:
                class: ray.tune.choice
                kwargs:
                categories: ["Adam", "SGD"]

```

**Sample Architecture**

Here we provide the base architecture and training configuration whose values
are going to be overwritten during hyperparameter optimization by
`config_mapper` the argument of
`core_pipelines.pipelines.train_dl_model.create_pipeline`

```yaml
  architecture:
    layer_01:
      n_block: 1
      class: "tensorflow.keras.layers.Dense"
      kwargs:
        units: 20
        activation: "relu"
    layer_02:
      n_block: 1
      class: "tensorflow.keras.layers.BatchNormalization"
      kwargs: {}
    layer_03:
      n_block: 1
      class: "tensorflow.keras.layers.Dropout"
      kwargs:
        rate: 0.1
    layer_04:
      n_block: 1
      class: "tensorflow.keras.layers.Dense"
      kwargs:
        units: 10
        activation: "relu"
    layer_05:
      n_block: 1
      class: "tensorflow.keras.layers.BatchNormalization"
      kwargs: {}
    layer_06:
      n_block: 1
      class: "tensorflow.keras.layers.Dropout"
      kwargs:
        rate: 0.1
    layer_07:
      n_block: 1
      class: "tensorflow.keras.layers.Dense"
      kwargs:
        units: 1
        activation: swish
  optimizer:
    class: "tensorflow.keras.optimizers.Adam"
    kwargs:
      learning_rate:
        class: tensorflow.optimizers.schedules.CosineDecay # default adam lr = 0.001
        kwargs:
          initial_learning_rate: 0.001
          decay_steps: 10
          alpha: 0.000001
```

**config_mapper**

As explained [here](#the-config_mapper), this function needs to take both a
hyperparameter search candidate and the dl params and return params that
represent the configuration for said hyperparameter search candidate. i.e., it
needs to take a sample from the hyperparameter search space (together with the
dl params) and return and instance where the relevant parts in the above
configuration are overwritten.

```python
def my_config_mapper(params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Use hyperparameter search space sample `config` to update `params`.

    Args:
        params (Dict[str, Any]): dl train pipeline parameters.
        config (Dict[str, Any]): sample from hyperparameter search.

    Returns:
        Dict[str, Any]: Updated parameters using values from `config`.
    """
    params["optimizer"]["class"] = f"tensorflow.keras.optimizers.{config['optimizer']}"
    blocks = _get_block(
        0, config["dropout1"], config["internal_activation"], int(config["units1"])
    )
    last_idx = 3
    if config["two_layers"]:
        new_block = _get_block(
            3, config["dropout2"], config["internal_activation"], int(config["units2"])
        )
        blocks = {**blocks, **new_block}
        last_idx = 6
    final_layer = {
        f"layer_{last_idx}": {
            "n_block": 1,
            "class": "tensorflow.keras.layers.Dense",
            "kwargs": {"units": 1, "activation": config["final_activation"]},
        }
    }
    blocks = {**blocks, **final_layer}
    lr = config.get("initilearning_rate", 0.005)
    params["optimizer"]["kwargs"]["learning_rate"]["kwargs"][
        "initial_learning_rate"
    ] = lr
    params["architecture"] = blocks
    return params


def _get_block(i, dropout: float, activation: str, units: int) -> Dict[str, Any]:
    return {
        f"layer_{i}": {
            "n_block": 1,
            "class": "tensorflow.keras.layers.Dense",
            "kwargs": {"units": units, "activation": activation},
        },
        f"layer_{i+1}": {
            "n_block": 1,
            "class": "tensorflow.keras.layers.BatchNormalization",
            "kwargs": {},
        },
        f"layer_{i+2}": {
            "n_block": 1,
            "class": "tensorflow.keras.layers.Dropout",
            "kwargs": {"rate": dropout},
        },
    }
```

Hence, with the above updates to the parameters file of the train dl pipeline
and instantiating the train_dl_pipeline, e.g., at the project pipeline registry,
with
`core_pipelines.pipelines.train_dl_model.create_pipeline(my_config_mapper)`,
we'll do hyperparameter search over the above search space.


## The `SamplingModelMixin` class
This module includes the `SamplingModelMixin` class, which implements a number of methods to aid sampling from probabilistic
models. Users are encouraged to use this in their derived/custom classes as a straightforward way to have access to these 
methods. For reference, please check the documentation and/or implementation of the `ProbabilisticKerasEstimator`.

## Data Sets
The deep learning pipeline expects two inputs - the dataset of features on which we'll be building our model and a tag dictionary object that has information about the different columns. It returns a model performance report, performance metrics, and a set of predictions on a hold out set.

### Inputs
The deep learning pipeline requires the following:
- `input`: The input dataset that should be cleaned.
- `td`: Tag dictionary object

### Outputs
The following output is created:
- `best_config`: Dictionary of best hyper-parameters for model
- `test_set_metrics`: Performance evaluation metrics from the test data
- `test_set_predictions`: Predictions from the test data.
- `train_set_feature_importance`: Permutation feature importances of the model from the training set. 
- `tune_results`: Results of hyper-parameter tuning trials
- `full_set_model`: The model with best hyper-parameters, fit on all of `train_dl_model.input`.

### Intermediate outputs
The train_dl_model pipeline has the following intermediate outputs:
- `ts_model`: Model fit on training set with best hyper-params, for evaluation on holdout test set.
- `train_set`: Training set that is used to build model
- `test_set`: Test set that is used for prediction


### Appendix: Some additional example config architectures
LSTM:
```yaml
architecture0:
    layer_01:
        class: "tensorflow.keras.layers.LSTM"
        kwargs:
            units: 12
            activation: "relu"
            return_sequences: True
    layer_02:
        class: "tensorflow.keras.layers.TimeDistributed"
        kwargs:
            layer:
                class: "tensorflow.keras.layers.Dense"
                kwargs:
                    units: 1
                    activation: 'relu'
```
Sequence to Sequence LSTM
``` yaml
architecture1:
    layer_01:
        class: "tensorflow.keras.layers.LSTM"
        kwargs:
            units: 12
            activation: "relu"
            return_sequences: True
    layer_02:
        class: "tensorflow.keras.layers.Dense"
        kwargs:
            units: 15
            activation: 'relu'
    layer_03:
        class: "tensorflow.keras.layers.TimeDistributed"
        kwargs:
            layer:
                class: "tensorflow.keras.layers.Dense"
                kwargs:
                    units: 15
                    activation: 'relu'
                    units: 1
``` 
Bidirectional LSTM
``` yaml
architecture2:
    layer_01:
        class: 'tensorflow.keras.layers.Bidirectional'
        n_repeat: 1
        kwargs:
            layer:
                class: "tensorflow.keras.layers.LSTM"
                kwargs:
                    units: 12
                    activation: "relu"
                    return_sequences: True
    layer_02:
        class: "tensorflow.keras.layers.Dense"
        n_repeat: 2
        kwargs:
            units: 15
            activation: 'relu'
    layer_03:
        n_block: 1
        class: "tensorflow.keras.layers.Dense"
        kwargs:
            bias_initializer:
                class: tensorflow.keras.initializers.RandomUniform
                kwargs:
                    minval: 800
                    maxval: 900
            units: 10
            activation: "relu"

    layer_03:
        class: "tensorflow.keras.layers.TimeDistributed"
        n_repeat: 1
        kwargs:
            layer:
                class: "tensorflow.keras.layers.Dense"
                kwargs:
                    units: 15
                    activation: 'relu'
                    units: 1
```                    
