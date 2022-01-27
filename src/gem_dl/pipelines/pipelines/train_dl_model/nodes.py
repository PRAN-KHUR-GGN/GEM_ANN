# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
"""
Nodes for building neural net models
"""
# pylint: disable=no-name-in-module, too-many-locals
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Mapping, Tuple, Union

import pandas as pd
import numpy as np
import ray
from optimus_core.core import utils
from optimus_core.core.tag_management import TagDict
from optimus_core.core.transformers import SelectColumns, SklearnTransform
from ray import tune
from sklearn.pipeline import Pipeline as SklearnPipeline
from tensorflow.python.keras.wrappers.scikit_learn import BaseWrapper

from ..train_model.nodes import feature_importance
from .keras_estimator import wrap_class

logger = logging.getLogger(__name__)


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


def _make_layer(key: str, config_dict: Dict) -> Any:
    """
    Load a tf.keras layer object based on specification in the config dictionary.

    Args:
        key:
        config_dict:

    Returns:

    """
    # Some layers (like Bidirectional) directly wrap other layers
    key_string = "|".join(config_dict[key]["kwargs"].keys())
    config_dict = deepcopy(config_dict)
    if "layer" in config_dict[key]["kwargs"].keys():
        wrapped_layer_conf = config_dict[key]["kwargs"]["layer"]
        wrapped_layer = utils.load_obj(wrapped_layer_conf["class"])(
            **wrapped_layer_conf["kwargs"]
        )
        config_dict[key]["kwargs"]["layer"] = wrapped_layer
    elif "initializer" in key_string:
        # Find initializer config, for recurrent_initializer,
        # bias_initializer, weight_initializer etc
        # this can be an optional config settings
        initializer_conf = {
            init_key: config_dict[key]["kwargs"][init_key]
            for init_key in config_dict[key]["kwargs"].keys()
            if "initializer" in init_key
        }
        # load initializers where needed with kwargs
        initializers_dict = {
            initializer: utils.load_obj(initializer_conf[initializer]["class"])(
                **initializer_conf[initializer]["kwargs"]
            )
            for initializer in initializer_conf
        }
        # Update config to have argument: initializer object k, v pairs
        for init_key in initializers_dict:
            config_dict[key]["kwargs"][init_key] = initializers_dict[init_key]

    layer = utils.load_obj(config_dict[key]["class"])(**config_dict[key]["kwargs"])
    return layer


def create_sequential_model(params: dict, model=None) -> Any:
    """
    Build a keras model using the Sequential api.
    Once compiled, keras models have fit and predict methods

    Args:
        params: Dictionary with model config parameters from parameters.yml
        model: Keras model object

    Returns: A compiled keras model.

    """
    # see https://docs.ray.io/en/latest/using-ray-with-tensorflow.html
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    input_shape = params["input_shape"]
    layers = params["architecture"]

    if not layers:
        raise RuntimeError("You've passed a layers config without layers!")

    # Sort and cast so that order isn't lost since layers is a dict.
    sorted_keys = sorted(list(layers.keys()))

    # Initiate a Keras Sequential model if not provided
    if not isinstance(model, tf.keras.Sequential):
        model = tf.keras.Sequential()
        y = tf.keras.layers.Input(shape=input_shape)
        model.add(y)

    for item in sorted_keys:
        for _ in range(layers[item].get("n_block", 1)):
            model.add(_make_layer(item, layers))

    loss_fn = [
        utils.load_obj(params["loss"][loss]["class"])(**params["loss"][loss]["kwargs"])
        for loss in params["loss"].keys()
    ]

    optimizer = utils.load_obj(params["optimizer"]["class"])(
        **params["optimizer"]["kwargs"]
    )
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=params["metrics"])

    return model


def create_lattice_model(params: dict) -> Any:
    """
    Build a Lattice Keras model using the Sequential api.
    Once compiled, Lattice models have fit and predict methods

    Args:
        params: Dictionary with model config parameters from parameters.yml

    Returns: A compiled Lattice model.

    """
    # see https://docs.ray.io/en/latest/using-ray-with-tensorflow.html
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    import tensorflow_lattice as tfl  # pylint: disable=import-outside-toplevel

    input_shape = params["input_shape"]
    layers = params["architecture"]
    calib_layers = params["calib_layers"]

    if not layers:
        raise RuntimeError("You've passed a layers config without layers!")

    model = tf.keras.Sequential()
    y = tf.keras.layers.Input(shape=input_shape)
    model.add(y)

    # Add the calibration layer for each feature to the model
    calibrators_layers = []
    for item in list(calib_layers):
        calibrator = _make_layer(item, calib_layers)
        calibrators_layers.append(calibrator)

    combined_calibrators = tfl.layers.ParallelCombination(calibrators_layers)
    model.add(combined_calibrators)

    # Define the remainder of the model architecture from the config
    model = create_sequential_model(params, model)

    return model


def _grab_monotonicity_constraints(td_frame: pd.DataFrame, feat_cols):
    """
    Grab the monotonicities and convexities to be enforced in a Lattice model.

    Args:
        td_frame: Tag dictionary dataframe
        feat_cols: List of features columns used in model

    Returns: two dictionaries, with the monotonicities and convexities to be
    enforced for all the features respectively
    """

    # Grab the monotonicities from the tag dictionary
    monotonicity_dict = {
        f: td_frame.loc[td_frame["tag"] == f, "monotonicity"]
        .replace(-1, "decreasing")
        .replace(1, "increasing")
        .fillna("none")
        .values[0]
        for f in feat_cols
    }

    # Also grab the convexity from the tagdict, if provided
    if "convexity" in td_frame.columns:
        convexity_dict = {
            f: td_frame.loc[td_frame["tag"] == f, "convexity"]
            .replace(-1, "concave")
            .replace(1, "convex")
            .fillna("none")
            .values[0]
            for f in feat_cols
        }
    else:
        convexity_dict = {f: "none" for f in feat_cols}

    return monotonicity_dict, convexity_dict


def _update_lattice_layer_kwargs(params, feat_cols, mono_dict, input_data):
    """
    Update the lattice layer kwargs in the model architecture.

    Args:
        params: Dictionary with model config parameters from parameters.yml
        feat_cols: List of features columns used in model
        mono_dict: Dictionary with monotonicity enforced for model features
        input_data: Input data used for model training

    Returns: Updated model architecture with kwargs added to lattice layer
    """
    # Grab the existing model architecture
    model_arch = params["architecture"]

    # Find the lattice layer in the architecture defined
    lattice_layer = [
        k
        for k in model_arch.keys()
        if model_arch[k]["class"] == "tensorflow_lattice.layers.Lattice"
    ][0]

    # Define the kwargs for the lattice layer in the architecture
    model_arch[lattice_layer]["kwargs"] = {}
    model_arch[lattice_layer]["kwargs"]["lattice_sizes"] = [
        params["lattice_size"]
    ] * len(feat_cols)
    model_arch[lattice_layer]["kwargs"]["monotonicities"] = [
        v.replace("decreasing", "increasing") for v in mono_dict.values()
    ]
    model_arch[lattice_layer]["kwargs"]["output_min"] = input_data[
        params.get("target")
    ].min()
    model_arch[lattice_layer]["kwargs"]["output_max"] = input_data[
        params.get("target")
    ].max()

    return model_arch


def _update_lattice_config(kwargs, td, input_data):
    """
    Update the model architecture config used to instantiate the neural
    network model, if building a lattice model. Add information for
    creating the piecewise calibration layer needed for lattice models.

    Args:
        kwargs: Dictionary containing info for instantiating keras model
        td: Tag dict
        input_data: Input data used for model training

    Returns:
        Updated kwargs dictionary with info for calibration layer
    """

    params = kwargs["params"]
    feat_cols = params.get("features", "model_feature")
    feat_cols = td.select(feat_cols)

    # If building a lattice model, ensure a Lattice layer is in the architecture
    model_arch = params["architecture"]
    all_classes = [model_arch[key]["class"] for key in model_arch]
    if "tensorflow_lattice.layers.Lattice" not in all_classes:
        raise RuntimeError(
            "No Lattice layers provided in architecture. Atleast "
            "one Lattice layer is needed in a lattice model"
        )

    # Grab the monotonicities and convexities to be enforced from the tagdict
    td_frame = td.to_frame()
    mono_dict, conv_dict = _grab_monotonicity_constraints(td_frame, feat_cols)

    # Define the calibration layers for all the feature columns
    params["calib_layers"] = {
        f"layer_{f}": {
            "class": "tensorflow_lattice.layers.PWLCalibration",
            "kwargs": {},
        }
        for f in feat_cols
    }
    for feat in feat_cols:
        calib_layer_kwargs = params["calib_layers"][f"layer_{feat}"]["kwargs"]
        calib_layer_kwargs["input_keypoints"] = np.quantile(
            input_data[feat], np.linspace(0.0, 1.0, num=5)
        )
        calib_layer_kwargs["output_min"] = 0
        calib_layer_kwargs["output_max"] = params["lattice_size"]

        calib_layer_kwargs["monotonicity"] = mono_dict[feat]
        calib_layer_kwargs["convexity"] = conv_dict[feat]

    params["architecture"] = _update_lattice_layer_kwargs(
        params, feat_cols, mono_dict, input_data
    )

    kwargs["params"] = params

    return kwargs


def _create_keras_config(preproc_pipeline, input_data, params, td):
    """
    Create the config needed for instantiating our neural network,
    which is wrapped in the KerasEstimator class.

    Args:
        preproc_pipeline:
        input_data:
        params:
        td:
    Returns:

    """
    n_feat = preproc_pipeline.fit_transform(input_data.head()).shape[1]
    input_shape = (n_feat,)
    if params.get("data_reformatter", {}).get("kwargs", {}).get("length", {}):
        n_times = params.get("data_reformatter", {}).get("kwargs", {}).get("length", {})
        input_shape = (n_times, n_feat)

    custom_objects = params.get("custom_objects", {})
    if custom_objects:
        for k, v in custom_objects.items():
            custom_objects[k] = utils.load_obj(v)
        params["custom_objects"] = custom_objects
    params["input_shape"] = input_shape
    kwargs = {
        "params": params,
    }
    # Model build_fn
    build_fn = params.get("build_fn")
    if build_fn is not None:
        build_fn = utils.load_obj(build_fn)
        kwargs["build_fn"] = build_fn

    # Update the model architecture if building a Lattice model
    if "lattice" in params.get("build_fn"):
        kwargs = _update_lattice_config(kwargs, td, input_data)

    estimator_dict = {
        "estimator": {"class": params["estimator"]["class"], "kwargs": kwargs}
    }

    return estimator_dict


def create_keras_model(params: Dict, td: TagDict, data: pd.DataFrame):
    """
    Creates a sklearn model pipeline based on the estimator and adds
    the desired transformers. This is where things like imputation,
    scaling, feature selection, and dynamic feature generation should plug in.

    Args:
        td: TagDict
        data: input data
        params: dictionary of model params
    Returns:
        sklearn model pipeline with transformers
    """

    # Transformer which reduces the model input to the
    # relevant features
    preproc_pipeline = _create_preprocess_pipeline(td, params)

    estimator_dict = _create_keras_config(preproc_pipeline, data.copy(), params, td)

    keras_estimator = load_keras_estimator(
        estimator_dict,
        params.get("data_reformatter"),
        params.get("probabilistic", False),
    )
    model = SklearnPipeline(preproc_pipeline.steps + [("estimator", keras_estimator)])

    return model


def load_keras_estimator(
    estimator_dict: Dict[str, Any],
    data_reformatter: Union[None, Dict[str, Any]] = None,
    probabilistic: bool = False,
) -> BaseWrapper:
    """Loads a estimator object based on given parameters.

    Args:
        estimator_dict (Dict[str, Any]): estimator config.
        data_reformatter (Union[None, Dict[str, Any]], optional): data reformatter
        config. Defaults to None.
        probabilistic (bool, optional): wether to add probabilistic functionality
        or not.

    Returns:
        BaseWrapper: estimator with the appropiate extensions
    """
    model_class = estimator_dict["estimator"]["class"]
    model_kwargs = estimator_dict["estimator"]["kwargs"]
    keras_estimator_class = utils.load_obj(model_class)
    assert hasattr(keras_estimator_class, "fit"), "Model object must have a .fit method"
    assert hasattr(
        keras_estimator_class, "predict"
    ), "Model object must have a .predict method"
    return wrap_class(keras_estimator_class, data_reformatter, probabilistic)(
        **model_kwargs
    )


def _create_preprocess_pipeline(td: TagDict, params: Mapping):
    """
    Creates a sklearn preproc pipeline with the
    the desired transformers. This is where things like imputation,
    scaling, should plug in.

    If a user would like to do pre-selection via the TagDict, they just
    need to specify in params the column name marking the features for
    their model.

    If the user has specified a list of features in params, this list will be
    used directly.

    Args:
        td: tag dictionary
        params: train_model params
    Returns:
        sklearn model pipeline with transformers
    """

    # Transformer which reduces the model input to the
    # relevant features
    # Get feature selectors
    feat_cols = params.get("features", "model_feature")
    if isinstance(feat_cols, str):
        feat_cols = td.select(feat_cols)
    column_selector = SelectColumns(feat_cols)

    transformer_steps = []
    if "transformers" in params:
        transformer_steps = [
            load_transformer(transformer_config)
            for transformer_config in params.get("transformers")
        ]
    model = SklearnPipeline([("select_columns", column_selector)] + transformer_steps)

    return model


def load_transformer(
    transformer_config: Mapping[str, Any]
) -> Tuple[str, SklearnTransform]:
    """
    Instantiates a transformer object from a configuration dictionary.

    Args:
        transformer_config: Configuration describing the transformer,
         including its path.
    Returns:
        An instantiated transformer object

    """
    required_keys = ["class", "kwargs", "name"]
    if not all([k in transformer_config for k in required_keys]):
        raise KeyError(
            f"Your transformer_config {transformer_config} is missing"
            f" (at least) one of the following keys {required_keys}"
        )
    transformer_path = transformer_config.get("class")
    kwargs = transformer_config.get("kwargs", {})
    step_name = transformer_config.get("name", "transformer_step")

    # selector_path will be None if no transformers are in params
    if transformer_path is not None:
        transformer = utils.load_obj(transformer_path)(**kwargs)

    # wrap sklearn style transformers to maintain pandas dfs
    # selector is True, False, or None. If none, returned unwrapped transformer
    transformer = SklearnTransform(transformer)
    return step_name, transformer


def fit_best_keras_model(
    params: Dict,
    data: pd.DataFrame,
    model: SklearnPipeline,
    wait_on=None,  # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """
    Args:
        params: dictionary of params
        data: input data
        model: sklearn pipeline with estimator and transformers
        wait_on: enforce execution order
    Returns:
        Tuple of trained sklearn compatible model and feature importances
    """
    target_col = params.get("target")
    target = data[target_col]
    if "ray" in params["fit"].get("callbacks", {}):
        params["fit"]["callbacks"].pop("ray")
    # # strictly speaking, selection features should not be necessary
    # # as this is done by the model transformer. However, we do it
    # # regardless to reduce our memory footprint and protect against
    # # accidental removal of the transformer
    feat_cols = model[:-1].transform(data.head()).columns.tolist()
    feature_df = data[feat_cols]
    model = train_model(params, data, model)
    feature_importances = feature_importance(model, feature_df, target)
    return dict(model=model, feature_importance=feature_importances)


def predict_bounds(data: pd.DataFrame, model: SklearnPipeline) -> pd.DataFrame:
    """adds columns `prediction_lower_bound` and `prediction_upper_bound` to
    `data` with the lower and upper bounds respectively of 100 random prediction
    samples.

    Args:
        data (pd.DataFrame): input data
        model (SklearnPipeline): model whose estimator has a `predict_interval`
        method.

    Returns:
        pd.DataFrame: input data with added columns for the bounds
    """
    x = model[:-1].transform(data)
    model.predict(x.head())
    model = model[-1]
    bounds = model.predict_interval(x, n_samples=100)
    data["prediction_lower_bound"], data["prediction_upper_bound"] = (
        bounds[:, 0],
        bounds[:, 1],
    )
    return data


def train_model(
    params: Dict[str, Any], data: pd.DataFrame, model: SklearnPipeline
) -> SklearnPipeline:
    """trains model with the provided data and parameters.

    Args:
        params (Dict[str, Any]): parameters for training
        data (pd.DataFrame): input data
        model (SklearnPipeline): model to be trained

    Returns:
        SklearnPipeline: trained model
    """
    train_max = int(params["split"]["train_split_fract"] * data.shape[0])
    train = data.iloc[:train_max, :]
    val = data.iloc[train_max:, :]
    callbacks_list = [
        utils.load_obj(callback_dict["class"])(**callback_dict["kwargs"])
        for _, callback_dict in params["fit"].get("callbacks", {}).items()
    ]
    target_col = params.get("target", "model_target")
    target = train[target_col]
    model.fit(
        train,
        target,
        estimator__validation_data=(model[:-1].fit_transform(val), val[target_col]),
        estimator__callbacks=callbacks_list,
        estimator__epochs=params["fit"]["epochs"],
        estimator__batch_size=params["fit"].get("batch_size"),
    )
    return model


def create_keras_trainer(
    td: TagDict,
    data: pd.DataFrame,
    params: Dict,
    config_mapper: Callable = map_config_params,
) -> Callable:
    """
    Returns a callable function that can be later optimized.

    It's meant to set all of the parameters (which don't change from trial to trial).

    Args:
        td: TagDictionary
        data: Training + Val data
        params: Dictionary of params + config
        config_mapper: Maps tune sets of params into train params

    Returns:

    """

    def keras_trainer(config):
        model = create_best_model(config, data, params, td, config_mapper)
        train_model(params, data, model)

    return keras_trainer


def tune_keras_estimator(tune_params: Dict, keras_trainer: Callable):
    """
    Leveraging `ray.tune` for distributed hyper-parameter tuning.

    This function creates a tuning function that creates and begins
    training a configurable neural network model. Once the best configuration
    of hyper-parameters is found, it builds a model with those parameters, and
    returns it along with the best configuration and a dataframe log of the
    details behind all of the trials.

    Some schedulers and search algorithms available within `tune` require
    additional keyword arguments which are not straight-forward expressed
    within a parameters.yaml config file. Concretely, when using
    population based training,, the scheduler requires the specification
    of hyperparam_mutations, which expects a dict of param: Callable
    key:value pairs. The `ray` package is a convenient interface to many
    other packages for hyper-parameter tuning, the user is encouraged to
    check out https://docs.ray.io/en/master/tune/api_docs/suggestion.html
    and https://docs.ray.io/en/master/tune/api_docs/schedulers.html for
    more information.

    By default, if no arguments for a search_alg or scheduler are present
    within config, default to random/grid search.

    Args:
        tune_params: A dictionary of parameters for training and fitting our DL model
        keras_trainer: Function to be passed to `ray.tune`. Likely output of

        config_mapper: dictionary mapping params to config param name.

    Returns: dictionary with keys:
        - best_config: Dict/JSON of best configurations.
        - tune_results: Dataframe of tuning configurations and performance results.
    """

    analysis = tune.run(keras_trainer, **tune_params)
    best_config = analysis.get_best_config(
        metric=tune_params.get("metric"), mode=tune_params.get("mode")
    )
    return dict(best_config=best_config, tune_results=analysis.dataframe())


def create_best_model(
    best_config: Dict,
    data: pd.DataFrame,
    params: Dict,
    td: TagDict,
    config_mapper: Callable = map_config_params,
) -> SklearnPipeline:
    """Takes the best configuration from hyperparameter tunning and builds
    a model with it.

    Args:
        best_config (Dict): configuration in the format returned by hyperparameter
        tunning
        data (pd.DataFrame): input data
        params (Dict): dictionary of model params
        td (TagDict): tag dictionary
        config_mapper (Callable, optional): Maps tune sets of params into train param.
        Defaults to `map_config_params`

    Returns:
        SklearnPipeline: [description]
    """
    params = config_mapper(params, best_config)
    best_model = create_keras_model(params, td, data)
    return best_model


def prepare_tune_params(tune_params):
    """
    Prepare dictionary of params for usage with ray[tune].

    Args:
        tune_params: Input dictionary of params

    Returns: Params with objects replaced by loaded objects where
    required.

    """
    # Add documentation for where this could be replaced by teams.
    try:
        if tune_params.get("scheduler") is not None:
            scheduler = utils.load_obj(tune_params["scheduler"]["class"])(
                **tune_params["scheduler"].get("kwargs", {})
            )
            tune_params["scheduler"] = scheduler
    except ray.tune.error.TuneError:
        raise Exception(
            "Please double check that the scheduler you "
            "are using has all of the appropriate "
            "arguments, either defined in "
            "`tune_keras_estimator` or in parameters.yml"
        )
    try:
        if tune_params.get("search_alg") is not None:
            search_alg = utils.load_obj(tune_params["search_alg"]["class"])(
                **tune_params["search_alg"].get("kwargs", {})
            )
            tune_params["search_alg"] = search_alg

    except (ray.tune.error.TuneError, ImportError):
        raise Exception(
            "Please double check that the search algorithm you "
            "would like to use is a) installed, b) has the "
            "appropriate arguments, either defined in code at  "
            "`tune_keras_estimator` or in parameters.yml"
        )
    # callbacks list
    if tune_params.get("callbacks", []):
        for idx, callback_config in enumerate(tune_params.get("callbacks")):
            tune_params["callbacks"][idx] = utils.load_obj(callback_config["class"])(
                **callback_config.get("kwargs", {})
            )
    # create search config space
    if tune_params.get("config", {}):
        tune_params["config"] = _create_search_space(tune_params.get("config", {}))
    return tune_params


def _create_search_space(search_space_config):
    """
    Within search_space, we've defined the parameter grid we'll be optimizing.
    Let's load the appropriate samplers.

    Args:
        search_space_config:

    Returns:

    """
    for k, v in search_space_config.items():
        search_space_config[k] = utils.load_obj(v["class"])(**v["kwargs"])
    return search_space_config
