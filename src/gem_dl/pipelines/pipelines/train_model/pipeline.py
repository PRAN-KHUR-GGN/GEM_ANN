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
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    add_transformers,
    create_predictions,
    load_estimator,
    retrain_model,
    train_model,
    verify_selected_controls,
    drop_any_nan,
)
from .reports import generate_performance_report, generate_performance_report_figures
from ..split.nodes import split_data


def create_pipeline():
    """
    Using this pipeline expects users to have a parameters file that includes
    a block like ::

        train_model:
            target: &train_model_target 'outp_quantity'
            features: 'model_feature'
            split:
                type: frac
                train_split_fract: 0.7 # iff `type: frac`
                datetime_val: !!timestamp '2020-04-26 03:59:59' # iff `type: date`
                datetime_col: "timestamp"
            transformers:
                # Specify list of dicts, each holding specification and name of a
                # transformer, feature selection or not
                - class: sklearn.feature_selection.SelectKBest
                  kwargs:
                      score_func: sklearn.feature_selection.mutual_info_regression
                      k: 7
                  name: mutual_information_selector
                  selector: True
            estimator:
                class: sklearn.neural_network.MLPRegressor
                kwargs:
                    random_state: 42
                    activation: relu
                    hidden_layer_sizes: 15
            cv:
                class: sklearn.model_selection.KFold
                kwargs:
                    n_splits: 5
                    random_state: null  # not needed without shuffle
            tuner:
                class: sklearn.model_selection.GridSearchCV
                kwargs:
                    n_jobs: -1
                    refit: mae
                    verbose: True
                    param_grid:
                        estimator__hidden_layer_sizes: [[15], [15,3]]
                        estimator__learning_rate: ['constant', 'adaptive']
                        estimator__activation: ['identity', 'relu']
                    scoring:
                        mae: neg_mean_absolute_error
                        rmse: neg_root_mean_squared_error
                        r2: r2
            report:
                output_path: .
                report_name: performance_report
                target: *train_model_target
                timestamp: True
                title: "Model Performance Report"
                author: "OptimusAI"
                subject: "OptimusAI Performance Report"
                shap:
                    explainer: shap.KernelExplainer
                    kwargs: {}


    Returns:

    """
    return pipeline(
        pipe=Pipeline(
            [
                node(
                    drop_any_nan,
                    dict(params="params:train_model", data="input", td="td"),
                    "input_dropna",
                    name="drop_any_nan",
                ),
                node(
                    split_data,
                    dict(params="params:train_model.split", data="input_dropna"),
                    dict(train="train_set", test="test_set"),
                    name="split_data",
                ),
                node(
                    load_estimator,
                    dict(params="params:train_model"),
                    "estimator",
                    name="load_estimator",
                ),
                node(
                    add_transformers,
                    dict(td="td", estimator="estimator", params="params:train_model"),
                    "estimator_pipeline",
                    name="add_transformers",
                ),
                node(
                    train_model,
                    dict(
                        params="params:train_model",
                        data="train_set",
                        model="estimator_pipeline",
                    ),
                    dict(
                        model="train_set_model",
                        cv_results="train_set_cv_results",
                        feature_importance="train_set_feature_importance",
                    ),
                    name="train_model",
                ),
                node(
                    verify_selected_controls,
                    dict(data="train_set", td="td", tuned_model="train_set_model"),
                    None,
                    name="log_no_controls_selected",
                ),
                node(
                    create_predictions,
                    dict(
                        params="params:train_model",
                        data="train_set",
                        model="train_set_model",
                    ),
                    dict(
                        predictions="train_set_predictions", metrics="train_set_metrics"
                    ),
                    name="create_train_predictions",
                ),
                node(
                    create_predictions,
                    dict(
                        params="params:train_model",
                        data="test_set",
                        model="train_set_model",
                    ),
                    dict(
                        predictions="test_set_predictions", metrics="test_set_metrics"
                    ),
                    name="create_test_predictions",
                ),
                node(
                    retrain_model,
                    dict(
                        params="params:train_model",
                        model="train_set_model",
                        data="input_dropna",
                    ),
                    "model",
                    name="retrain_model",
                ),
                node(
                    generate_performance_report_figures,
                    dict(
                        params="params:train_model.report",
                        feature_importance="train_set_feature_importance",
                        train_set_predictions="train_set_predictions",
                        test_set_predictions="test_set_predictions",
                        model="model",
                        test_set_metrics="test_set_metrics",
                        train_set_metrics="train_set_metrics",
                        static_features="input_dropna",
                    ),
                    "report_figures",
                    name="generate_performance_report_figures",
                ),
                node(
                    generate_performance_report,
                    dict(
                        params="params:train_model.report",
                        figures_dict="report_figures",
                    ),
                    None,
                    name="generate_performance_report",
                ),
            ]
        )
    )
