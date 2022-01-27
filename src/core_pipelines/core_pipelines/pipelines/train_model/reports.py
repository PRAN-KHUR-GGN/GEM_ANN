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

import logging
from typing import Dict

import pandas as pd
from .nodes import create_model_summary
from sklearn.pipeline import Pipeline as SklearnPipeline

from ..pyplot_charts.nodes import (
    plot_pandas_df,
    plot_timeline,
    plot_bar,
    plot_actual_vs_predicted,
    plot_actual_vs_residual,
    plot_string,
    plot_partial_dependency,
)
from ..pyplot_reports.nodes import generate_pdf

logger = logging.getLogger(__name__)


def generate_performance_report_figures(
    params: Dict,
    feature_importance: pd.DataFrame,
    train_set_predictions: pd.DataFrame,
    test_set_predictions: pd.DataFrame,
    model: SklearnPipeline,
    test_set_metrics: pd.DataFrame,
    train_set_metrics: pd.DataFrame,
    static_features: pd.DataFrame,
    summary: pd.DataFrame = pd.DataFrame(),
):
    """
    Generate a figures for a Performance Report
    Args:
        params: parameters as expected by pyplot_reports:pdf and with shap args
        feature_importance: feature importance data
        train_set_predictions: predictions from train set
        test_set_predictions: predictions from test set
        model: trained model
        test_set_metrics: the test set metrics data
        train_set_metrics: the training set metrics data
        static_features: the input features
        summary: summary table of modeling pipeline

    Returns:
        Dictionary of matplotlib figures that were used to generate the report
    """

    target = params.get("target", "model_target")

    static_feature_list = model[:-1].transform(static_features.head())
    timestamp_col = params.get("timestamp_col", "timestamp")

    # Create a table containing a summary of the model input data set and model
    # performance
    summary = create_model_summary(
        train_set_metrics, test_set_metrics, static_features, model
    )

    pdf_pages = {
        "Model summary.png": plot_pandas_df(
            dict(figsize=[16, 9], title="Model summary"), summary
        )["pandas_df_plot"],
        "SKLearn Model.png": plot_string(
            dict(figsize=[16, 9], title="SkLearn Model Used"), model
        )["string_plot"],
        "Performance Test Metrics.png": plot_pandas_df(
            dict(
                figsize=[16, 9],
                title="Test Dataset Performance Metrics for Best Parameters",
            ),
            test_set_metrics,
        )["pandas_df_plot"],
        "Performance Train Metrics.png": plot_pandas_df(
            dict(
                figsize=[16, 9],
                title="Train Dataset Performance Metrics for Best Parameters",
            ),
            train_set_metrics,
        )["pandas_df_plot"],
        "Feature Importance Plot.png": plot_bar(
            dict(
                figsize=[16, 9],
                title="Feature Importance Plot",
                y="feature_importance",
                sort_by="feature_importance",
                annotate=True,
            ),
            feature_importance,
        )["bar_plot"],
        "Model Actual VS Predicted.png": plot_actual_vs_predicted(
            dict(
                figsize=[16, 9],
                title="Train Model Actual VS Predicted",
                prediction_col="prediction",
                target_col=target,
            ),
            test_set_predictions,
        )["actual_vs_predicted_plot"],
        "Model Actual VS Residual.png": plot_actual_vs_residual(
            dict(
                figsize=[16, 9],
                title="Train Model Actual VS Residual",
                prediction_col="prediction",
                target_col=target,
            ),
            test_set_predictions,
        )["actual_vs_residual_plot"],
        "Partial Dependence Plot.png": plot_partial_dependency(
            dict(
                figsize=[16, 9],
                title="Model PDP",
                ylabel=target,
                columns=static_feature_list,
            ),
            static_features,
            model,
        )["partial_dependency_plot"],
        "Train Timeline.png": plot_timeline(
            dict(
                figsize=[16, 9],
                title="Train Set Timeline",
                columns=[target, "prediction"],
                n=10000,
                timestamp_col=timestamp_col,
            ),
            train_set_predictions,
        )["timeline_plot"],
        "Test Timeline.png": plot_timeline(
            dict(
                figsize=[16, 9],
                title="Test Set Timeline",
                columns=[target, "prediction"],
                n=10000,
                timestamp_col=timestamp_col,
            ),
            test_set_predictions,
        )["timeline_plot"],
    }

    return pdf_pages


def generate_performance_report(params: Dict, figures_dict: Dict):
    """
    Write a pdf model performance report to disk. Path to saved file
    taken from params. Path specified in params at
    params['train_model']['report']['output_path'].
    Args:
        params: report generation params
        figures_dict: Dict of figures, keyed by values

    Returns: None

    """
    generate_pdf(params, *figures_dict.values())
