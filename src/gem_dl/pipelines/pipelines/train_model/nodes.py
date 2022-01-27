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
Nodes of the model training pipeline.
"""
import logging
from copy import deepcopy
from typing import Any, Dict, Mapping

import pandas as pd
from optimus_core.core import utils
from optimus_core.core.tag_management import TagDict
from optimus_core.core.transformers import SelectColumns, SklearnTransform
from optimus_core.core.transformers.feature_selection import SkLearnSelector
from sklearn.pipeline import Pipeline as SklearnPipeline
from xgboost import XGBRegressor

from .performance import generate_prediction_metrics, feature_importance
from .tuning import sklearn_tune, xgb_tune, grab_monotonicity
from ..reporting_html.utils import get_model_vars

logger = logging.getLogger(__name__)
quality_check_logger = logging.getLogger("quality_check_logger")


def load_estimator(params: dict, td: TagDict = None):
    """
    Loads a estimator object based on given parameters.
    Args:
        params: dictionary of parameters
        td: tag dictionary
    Returns:
        sklearn compatible model
    """
    model_class = params["estimator"]["class"]

    if td is not None:
        params = grab_monotonicity(params, td)
    model_kwargs = params["estimator"]["kwargs"]
    estimator = utils.load_obj(model_class)(**model_kwargs)
    assert hasattr(estimator, "fit"), "Model object must have a .fit method"
    assert hasattr(estimator, "predict"), "Model object must have a .predict method"
    return estimator


def load_transformer(transformer_config: Mapping[str, Any]) -> Any:
    """
    Instantiates a transformer object from a configuration dictionary.
    Assumes that the keyword argument `score_func` in the selector class arguments
    always expects a python function. This function will load the object at the path
    provided and pass it into the `score_func` kwarg.
    Assumes that the keyword argument `estimator` in the selector class arguments
    always expects an estimator class. This function will load the object at the path
    provided and use the `estimator_args` configuration entry as estimator class
    arguments. All values passed under `estimator_args` will not be passed into the
    selector, but are provided to the estimator only.

    transformer_config requires the following fields:
        * estimator: the import path of the selector.
        * kwargs: (Optional) keyword arguments to be passed to the selector.

    Args:
        transformer_config: Configuration describing the selector, including its path.
    Returns:
        An instantiated selector object

    Example:
        all_selectors = [
        _instantiate_selector(selector_params)
        for selector_params in all_selector_params]
        all_selectors-> list of tuples
        pipeline_steps = all_selectors + estimator
    """
    required_keys = ["class", "kwargs", "name", "selector"]
    if not all([k in transformer_config for k in required_keys]):
        raise KeyError(
            f"Your transformer_config {transformer_config} is missing"
            f" (at least) one of the following keys {required_keys}"
        )
    selector_path = transformer_config.get("class")
    kwargs = transformer_config.get("kwargs", {})
    step_name = transformer_config.get("name", "transformer_step")
    if "estimator" in kwargs:
        estimator_kwargs = kwargs["estimator"].pop("kwargs", {})
        estimator_class = kwargs["estimator"].pop("class", "")
        kwargs["estimator"] = utils.load_obj(estimator_class)(**estimator_kwargs)

    if "score_func" in kwargs:
        kwargs["score_func"] = utils.load_obj(kwargs["score_func"])

    # selector_path will be None if no transformers are in params
    selector = selector_path
    if selector_path is not None:
        selector = utils.load_obj(selector_path)(**kwargs)

    # wrap sklearn style transformers to maintain pandas dfs
    # selector is True, False, or None. If none, returned unwrapped transformer
    if transformer_config.get("selector"):
        selector = SkLearnSelector(selector)
    elif transformer_config.get("selector") is False:
        selector = SklearnTransform(selector)
    return step_name, selector


def add_transformers(td: TagDict, estimator: Any, params: Mapping):
    """
    Creates a sklearn model pipeline based on the estimator and adds
    the desired transformers. This is where things like imputation,
    scaling, feature selection, and dynamic feature generation should plug in.

    If a user would like to do pre-selection via the TagDict, they just
    need to specify in params the column name marking the features for
    their model.

    If the user has specified a list of features in params, this list will be
    used directly.

    Args:
        td: tag dictionary
        estimator: Instantiated model.
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
    model = SklearnPipeline(
        [("select_columns", column_selector)]
        + transformer_steps
        + [("estimator", estimator)]
    )
    return model


def train_model(
    params: dict, data: pd.DataFrame, model: SklearnPipeline
) -> Dict[str, Any]:
    """
    Args:
        params: dictionary of parameters
        data: input data
        model: sklearn pipeline with estimator and transformers
    Returns:
        Trained sklearn compatible model
        Hyper-parameter Tuning Search Results
        Feature importance
    """
    target_col = params.get("target", "model_target")
    target = data[target_col]

    estimator = model.named_steps["estimator"]
    if isinstance(estimator, XGBRegressor):
        logger.info("Tuning using `xgb_tune`.")
        tuned_model, cv_results_df = xgb_tune(params, data, target, model)
    else:
        logger.info("Tuning using `sklearn_tune`.")
        tuned_model, cv_results_df = sklearn_tune(params, data, target, model)

    importances = feature_importance(tuned_model, data, target)

    return dict(
        model=tuned_model, cv_results=cv_results_df, feature_importance=importances
    )


def verify_selected_controls(
    data: pd.DataFrame, td: TagDict, tuned_model: SklearnPipeline
):
    """
    Verification function, checks if control_variables are selected by the model
    after algorithmic feature_selection

    Args:
        data: Training dataset to evaluate on.
        td: Tag Dictionary detailing which inputs are control vs state variables.
        tuned_model: Tuned/fit predictive model.

    """
    # Check that there is at least 1 control which has been selected
    if isinstance(td, pd.DataFrame):
        _mask = td["tag_type"].apply(lambda x: x == "control")
        control_vars = set(list(td.loc[_mask, "tag"]))
    else:
        control_vars = set(td.select("tag_type", "control"))
    selected_vars = set(tuned_model[:-1].transform(data).columns.tolist())
    if control_vars.isdisjoint(selected_vars):
        logger.log(
            level=30,
            msg="There were no control variables which were"
            " selected! This can have effects on how"
            "well your model can be optimized.",
        )


def create_predictions(
    params: Dict, data: pd.DataFrame, model: SklearnPipeline
) -> Dict[str, pd.DataFrame]:
    """
    Creates model predictions for a given data set
    Args:
        params: dictionary of parameters
        data: input data
        model: sklearn pipeline with estimator and transformers
    Returns:
        predictions, metrics
    """
    prediction_col = "prediction"
    predictions = model.predict(data)
    target_col = params.get("target", "model_target")
    res_df = data.copy()
    if predictions.shape[0] != data.shape[0]:
        missing_rows = data.shape[0] - predictions.shape[0]
        res_df = res_df.iloc[missing_rows:, :]
    res_df[prediction_col] = predictions
    prediction_metrics_df = pd.DataFrame()
    prediction_metrics_df["opt_perf_metrics"] = generate_prediction_metrics(
        res_df, target_col, prediction_col
    )
    return dict(predictions=res_df, metrics=prediction_metrics_df)


def retrain_model(
    params: Dict, model: SklearnPipeline, data: pd.DataFrame
) -> SklearnPipeline:
    """
    Retraining the model object with the new dataset.
    Args:
        params: Dictionary of parameters
        model: sklearn pipeline with estimator and transformers
        data: input data

    Returns:
        retrained SklearnPipeline model

    """
    target_col = params.get("target", "model_target")
    target = data[target_col]

    model_to_retrain = deepcopy(model)
    model_to_retrain.fit(data, target)
    return model_to_retrain


def create_model_summary(
    train_set_metrics: pd.DataFrame,
    test_set_metrics: pd.DataFrame,
    features: pd.DataFrame,
    model: SklearnPipeline,
) -> pd.DataFrame:
    """
    Create a table containing a summary of the model input data set and model
    performance
    Args:
        train_set_metrics: training set metrics data
        test_set_metrics: test set metrics data
        features: input data set
        model: trained model pipeline
        params: params to be used

    Returns:
        dataframe containing the summary
    """
    # we want the variables that are actually used as features in the
    # pipeline
    sel_features = model[:-1].transform(features.head())
    model_type = type(model[-1]).__name__

    summary = {
        "# datapoints - train set": int(len(features)),
        "# features - train set": int(sel_features.shape[1]),
        "test MAE": test_set_metrics.loc["mae"][0],
        "test RMSE": test_set_metrics.loc["rmse"][0],
        "test R2": test_set_metrics.loc["r2"][0],
        "train R2": train_set_metrics.loc["r2"][0],
        "model type": model_type,
    }

    # check if input has timestamp column and get the timeframe range
    if "timestamp" in features:
        ts_min = min(features["timestamp"]).strftime("%Y-%m-%d")
        ts_max = max(features["timestamp"]).strftime("%Y-%m-%d")
        summary["time frame"] = f"({ts_min}, {ts_max})"

    model_summary = pd.DataFrame()
    # empty string column name since keys are self-explanatory
    model_summary[""] = pd.Series(summary)

    return model_summary


def drop_any_nan(params: Dict, data: pd.DataFrame, td: TagDict) -> pd.DataFrame:
    """
    Drops all rows that contain any nan.
    Args:
        td: tag dictionary to use
        params: parameters to be used
        data: input data
    Returns:
        data without any nan
    """
    n_samples_before = data.shape[0]

    # get model features and target variable
    model_vars = get_model_vars(params, td)
    if params.get("missing", "drop") == "drop":
        data = data.dropna(subset=model_vars)

    n_samples_after = data.shape[0]
    n_samples_dropped = n_samples_before - n_samples_after
    logger.info(
        f"Dropping {n_samples_dropped} samples with any NaN, reducing dataset from "
        f"{n_samples_before} samples to {n_samples_after} samples."
    )
    quality_check_logger.info(
        f"Dropping {n_samples_dropped} samples with any NaN, reducing dataset from "
        f"{n_samples_before} samples to {n_samples_after} samples."
    )

    return data
