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
from typing import Any, Mapping

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)
from sklearn.inspection import permutation_importance

from optimus_core.core.metrics import mean_absolute_percentage_error


logger = logging.getLogger(__name__)


def generate_prediction_metrics(
    data: pd.DataFrame, y_true_col: str, y_pred_col: str
) -> pd.Series:
    """
    Calculate various metrics:
     - MAE
     - RMSE
     - MSE
     - MAPE
     - R squared
     - Explained variance

    Args:
        data: Dataframe containing features and predictions
        y_true_col: the actual values column name
        y_pred_col: the predicted values column name

    Returns:
        A pandas series of metric values
    """
    y_true = data[y_true_col]
    y_pred = data[y_pred_col]
    metrics_vals = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mse": mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "var_score": explained_variance_score(y_true, y_pred),
    }
    return pd.Series(metrics_vals)


def feature_importance(
    model_pipeline: Any, data: pd.DataFrame, target: pd.Series
) -> Mapping:
    """
    Return pd.Series of feature importances.

    Will first try to return native/inherent feature importances from estimator.
        If not possible, will then return `permutation_importances`

    Args:
        model_pipeline: the model pipeline
        data: dataframe to transform and find selected features
        target: Model targets

    Returns:
        A Pandas Series of Feature Importances
    """
    model = model_pipeline.named_steps["estimator"]
    # Let's transform the data through every step except model
    features = model_pipeline[:-1].transform(data).columns.tolist()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        logger.log(
            level=30,
            msg=f"{type(model)}-type estimator does not have native"
            "`.feature_importances_` attribute. "
            "Returning permutation feature importances",
        )
        importances_dict = permutation_importance(model, data[features], target)
        importance = importances_dict["importances_mean"]
    importances = pd.Series(importance, features).to_frame()
    importances.columns = ["feature_importance"]

    return importances
