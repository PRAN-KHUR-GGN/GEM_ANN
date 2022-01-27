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
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.utils.validation import column_or_1d

from ..pyplot_charts.nodes import (
    plot_actual_vs_predicted,
    plot_actual_vs_residual,
    plot_bar,
    plot_pandas_df,
    plot_string,
    plot_timeline,
)
from ..pyplot_reports.nodes import generate_pdf

logger = logging.getLogger(__name__)


def generate_performance_report(params, report_pages):
    generate_pdf(params, *[v for _, v in report_pages.items()])


def generate_report_pages(
    params: Dict[str, Any],
    feature_importance: pd.DataFrame,
    train_set_predictions: pd.DataFrame,
    test_set_predictions: pd.DataFrame,
    model: SklearnPipeline,
    train_set_metrics: pd.DataFrame,
) -> Dict[str, Any]:
    """Code for actual generation of each page in the model performance
    report for the deep learning model.

    Args:
        params (Dict[str, Any]): report params
        feature_importance (pd.DataFrame): feature importance df
        train_set_predictions (pd.DataFrame):
        test_set_predictions (pd.DataFrame):
        model (SklearnPipeline):
        train_set_metrics (pd.DataFrame):

    Returns:
        Dict[str, Any]: dictionary of figures
    """

    target = params.get("target", "model_target")
    model._is_fitted = True
    tf_model_summary = get_tf_model_summary(model)
    pdf_pages = {
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
        "SKLearn Model.png": plot_string(
            dict(figsize=[16, 9], title="SkLearn Model Used"), model
        )["string_plot"],
        "Keras Model Summary.png": plot_string(
            dict(figsize=[16, 9], title="Model Summary"), tf_model_summary
        )["string_plot"],
        "Performance Metrics.png": plot_pandas_df(
            dict(
                figsize=[16, 9],
                title="Test Dataset Performance Metrics for Best Parameters",
            ),
            train_set_metrics,
        )["pandas_df_plot"],
        "Train Timeline.png": plot_timeline(
            dict(
                figsize=[16, 9],
                title="Train Set Timeline",
                columns=[target, "prediction"],
                n=10000,
            ),
            train_set_predictions,
        )["timeline_plot"],
        "Test Timeline.png": plot_timeline(
            dict(
                figsize=[16, 9],
                title="Test Set Timeline",
                columns=[target, "prediction"],
                n=10000,
            ),
            test_set_predictions,
        )["timeline_plot"],
    }
    return pdf_pages


def get_tf_model_summary(model):
    """
    FInd a string representation of a keras model.
    Args:
        model: sklearn pipeline object, containing a KerasEstimator.

    Returns: A string holding the model summary.

    """
    if isinstance(model, SklearnPipeline) and "estimator" in model.named_steps:
        # If the model is a KerasEstimator (and has the enable_prediction method to load
        # the keras model object itself, let's set the model to the underlying
        # keras model
        tf_model_summary = ""
        if hasattr(model.named_steps["estimator"], "model"):
            tmp = []
            model.named_steps["estimator"].model.summary(
                print_fn=lambda x: tmp.append(x)  # pylint:disable= unnecessary-lambda
            )
            tf_model_summary = "\n".join(tmp)
    return tf_model_summary


def generate_probabilistic_figures(
    params: Dict,
    train_set_predictions: pd.DataFrame,
    test_set_predictions: pd.DataFrame,
):
    """
    Generate figures relevant to a model which predicts uncertainty ranges
    Args:
        params: Dictionary of configuration params for reporting.
        train_set_predictions: Dataframe holding the features, target and prediction
        for the train set
        test_set_predictions: Dataframe holding the features, target and prediction
        for the test set

    Returns: Dictionary of figures displaying prediction intervals.
    """
    target = params.get("target", "model_target")
    # Calibration plot? -> Summary of % coverage for different hpd intervals?
    pdf_pages = {
        "Model Actual VS Predicted-Uncertainty.png": plot_actual_predicted_uncertainty(
            params=dict(
                figsize=[16, 9],
                title="Train Model Actual VS Predicted",
                prediction_col="prediction",
                upper_bound="prediction_upper_bound",
                lower_bound="prediction_lower_bound",
                target_col=target,
                plot_kwargs=dict(alpha=0.25, ecolor="red", elinewidth=3),
            ),
            data=test_set_predictions,
        )["actual_vs_predicted_plot"],
        "Train Timeline.png": plot_timeline(
            dict(
                figsize=[16, 9],
                title="Train Set Timeline",
                columns=[
                    target,
                    "prediction_lower_bound",
                    "prediction",
                    "prediction_upper_bound",
                ],
                n=10000,
            ),
            train_set_predictions,
        )["timeline_plot"],
        "Test Timeline.png": plot_timeline(
            dict(
                figsize=[16, 9],
                title="Test Set Timeline",
                columns=[
                    target,
                    "prediction_lower_bound",
                    "prediction",
                    "prediction_upper_bound",
                ],
                n=10000,
            ),
            test_set_predictions,
        )["timeline_plot"],
    }
    return pdf_pages


def plot_actual_predicted_uncertainty(params: Dict, data: pd.DataFrame):
    """
    Create a predicted vs actual plot, showing the range of potential predicted values.

    Args:
        params: Plotting params.
        data: Data holding the target , prediction, and upper/lower prediction bounds.

    Returns: Matplotlib figure holding predicted-vs actual plot
    """
    fig = plot_actual_vs_predicted(params, data)["actual_vs_predicted_plot"]
    ax = fig.axes[0]
    prediction, ub, lb, target = (
        params["prediction_col"],
        params["upper_bound"],
        params["lower_bound"],
        params["target_col"],
    )
    yerr = [
        (data[prediction] - data[lb]).values,
        (data[ub] - data["prediction"]).values,
    ]
    ax.errorbar(
        x=data[target].rename("Actual"),
        y=data[prediction].rename("Predicted"),
        ls="none",
        yerr=yerr,
        **params.get("plot_kwargs", {}),
    )

    ax.set_title(
        "Predicted vs Actual with Uncertainty - "
        "interval coverage = "
        f"{np.round(coverage_score(data[target], data[lb], data[ub]), decimals=3)}",
        fontsize=14,
    )
    fig.axes[0] = ax
    return {"actual_vs_predicted_plot": fig}


def coverage_score(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred_low: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred_up: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> float:
    """
    Effective coverage score obtained by the prediction intervals.
    Calculated as the faction of points lying in the prediction intervals.


    Args:
        y_true: True labels, shape (n,)
        y_pred_low: Lower bound of prediction intervals, shape (n,)
        y_pred_up: Upper bound of prediction intervals, shape (n,)

    Returns: Effective coverage of the predictive intervals

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([5, 9, 9.5, 5])
        >>> y_pred_low = np.array([4, 4, 9, 8.5])
        >>> y_pred_up = np.array([6, 12, 10, 12.5])
        >>> print(coverage_score(y_true, y_pred_low, y_pred_up))
        0.75


    """
    y_true = column_or_1d(y_true)
    y_pred_low = column_or_1d(y_pred_low)
    y_pred_up = column_or_1d(y_pred_up)
    coverage = ((y_pred_low <= y_true) & (y_pred_up >= y_true)).mean()
    return float(coverage)
