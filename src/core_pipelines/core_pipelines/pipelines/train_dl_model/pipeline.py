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
Model Training Pipeline
"""
from functools import partial
from typing import Callable

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_best_model,
    create_keras_trainer,
    fit_best_keras_model,
    map_config_params,
    prepare_tune_params,
    tune_keras_estimator,
)
from .reports import generate_report_pages
from ..split.nodes import split_data
from ..train_model.nodes import create_predictions, drop_any_nan


def create_pipeline(config_mapper: Callable = map_config_params) -> Pipeline:
    """
    This pipeline expects a params block with the following schema ::

        train_deeplearning:
            target: 'model_target'
            features: 'model_feature'
            split:
                type: frac
                train_split_fract: 0.8 # req. iff `type: frac`
                datetime_val: !!timestamp '2020-04-26 03:59:59' # req. iff `type: date`
                datetime_col: "timestamp"
            train_split_fract: 0.8
            transformers:
                - class: sklearn.preprocessing.StandardScaler
                  kwargs: {}
                  name: standard_scaler
            build_fn: |
                core_pipelines.pipelines.train_dl_model.nodes.create_sequential_model
            estimator:
                class: core_pipelines.pipelines.train_dl_model.KerasEstimator
            fit:
                epochs: 50
                callbacks:
                    tensorboard:
                        class: 'tensorflow.keras.callbacks.TensorBoard'
                        kwargs:
                            log_dir: "./logs"
                    es:
                        class: 'tensorflow.keras.callbacks.EarlyStopping'
                        kwargs:
                            patience: 100
                            monitor: 'val_loss'
                    ray: # ray[tune] keras integration callback
                        class: ray.tune.integration.keras.TuneReportCallback
                        kwargs:
                          metrics: "val_loss"
                          "on": 'epoch_end'
            metrics:
                - 'mean_absolute_error'
                - 'mean_absolute_percentage_error'
            architecture:
                layer_01:
                    n_block: 1
                    class: "tensorflow.keras.layers.Dense"
                    kwargs:
                        units: 20
                        activation: "relu"
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
            optimizer:
                class: 'tensorflow.keras.optimizers.Adam'
                kwargs:
                    learning_rate: 0.005 # default adam lr = 0.001
            loss:
                MeanAbsoluteError:
                    class: "tensorflow.keras.losses.MeanAbsoluteError"
                    kwargs: {} # Most losses don't accept extra arguments
            tune_params:
                metric: val_loss
                mode: min
                local_dir: data/07_model_output/ray_tune_results/ # store tune results
                fail_fast: True
                config:
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
                search_alg:
                    class: ray.tune.suggest.optuna.OptunaSearch
                    kwargs: {}
                # callbacks: - List[Dict]. Each dict-> class and kwarg: Dict[str,str]
                num_samples: 8
                verbose: 1 # level of messages to print
                stop:
                    training_iteration: 10
                resources_per_trial:
                    cpu: 1
                    gpu: 0
            report:
                output_path: data/08_reporting
                report_name: performance_report
                timestamp: True
                title: "Model Performance Report"
                author: "OptimusAI"
                subject: "OptimusAI Performance Report"
                shap:
                    explainer: shap.DeepExplainer
                    kwargs: {}



    Returns:

    """
    return pipeline(
        Pipeline(
            [
                node(
                    drop_any_nan,
                    dict(params="params:train_deeplearning", data="input", td="td"),
                    "input_dropna",
                    name="drop_any_nan",
                ),
                node(
                    split_data,
                    dict(params="params:train_deeplearning.split", data="input_dropna"),
                    dict(train="train_set", test="test_set"),
                    name="split_data",
                ),
                node(
                    partial(create_keras_trainer, config_mapper=config_mapper),
                    inputs=dict(
                        td="td", data="train_set", params="params:train_deeplearning",
                    ),
                    outputs="keras_trainer_func",
                    name="create_keras_trainable",
                ),
                node(
                    prepare_tune_params,
                    inputs="params:train_deeplearning.tune_params",
                    outputs="tune_params",
                    name="prepare_params_for_ray_tune",
                ),
                node(
                    tune_keras_estimator,
                    dict(keras_trainer="keras_trainer_func", tune_params="tune_params"),
                    dict(best_config="best_config", tune_results="tune_results"),
                    name="tune_keras_estimator",
                ),
                node(
                    partial(create_best_model, config_mapper=config_mapper),
                    inputs=dict(
                        best_config="best_config",
                        data="train_set",
                        params="params:train_deeplearning",
                        td="td",
                    ),
                    outputs="model",
                    name="create_best_model",
                ),
                node(
                    fit_best_keras_model,
                    dict(
                        params="params:train_deeplearning",
                        data="train_set",
                        model="model",
                    ),
                    dict(
                        model="ts_model",
                        feature_importance="train_set_feature_importance",
                    ),
                    name="train_model_on_train_set",
                ),
                node(
                    create_predictions,
                    dict(
                        params="params:train_deeplearning",
                        data="test_set",
                        model="ts_model",
                    ),
                    dict(
                        predictions="test_set_predictions", metrics="test_set_metrics"
                    ),
                    name="create_test_predictions",
                ),
                node(
                    create_predictions,
                    dict(
                        params="params:train_deeplearning",
                        data="train_set",
                        model="ts_model",
                    ),
                    dict(
                        predictions="train_set_predictions", metrics="train_set_metrics"
                    ),
                    name="create_train_predictions",
                ),
                node(
                    fit_best_keras_model,
                    dict(
                        params="params:train_deeplearning",
                        data="input_dropna",
                        model="model",
                        wait_on="test_set_metrics",
                    ),
                    dict(
                        model="full_set_model",
                        feature_importance="full_feature_importance",
                    ),
                    name="fit_optimal_model_full_data",
                ),
                node(
                    generate_report_pages,
                    dict(
                        params="params:train_deeplearning.report",
                        feature_importance="train_set_feature_importance",
                        train_set_predictions="train_set_predictions",
                        test_set_predictions="test_set_predictions",
                        model="full_set_model",
                        train_set_metrics="train_set_metrics",
                    ),
                    "report_figures",
                    name="generate_performance_report_figures",
                ),
            ]
        )
    )
