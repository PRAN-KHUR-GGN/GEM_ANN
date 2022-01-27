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
Model tuning procedures
"""
import logging
import numbers
from typing import Any, Dict, List, Union
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline as SklearnPipeline

from optimus_core.core import utils
from optimus_core.core.metrics import mean_absolute_percentage_error
from optimus_core.core.tag_management import TagDict

logger = logging.getLogger(__name__)


def get_cv_from_params(params: dict) -> BaseCrossValidator:
    """
    Creates CV Splitting Strategy from Params
    Default is 5-fold cross validation
    Args:
        params: dictionary of parameters
    Returns:
        Cross Validation Iterator
    """
    cv = params["cv"]
    if not isinstance(cv, numbers.Integral):
        cv = utils.load_obj(params["cv"]["class"])(**params["cv"]["kwargs"])
    return cv


def get_hp_from_params(params: dict, model: SklearnPipeline):
    """
    Instantiates Hyper-parameter Tuning strategy from Params.
    If scoring is specified in the params, it also adds MAPE in
    the scoring calculation.
    Args:
        params: dictionary of parameters
        model: sklearn pipeline with estimator and transformers
    Returns:
        Hyperparameter Search Strategy
    """
    cv = get_cv_from_params(params)

    args = params["tuner"]["kwargs"].copy()
    args["estimator"] = model
    args["cv"] = cv

    if "param_distributions" in args.keys():
        param_grid = args["param_distributions"]
        for k, v in param_grid.items():
            if isinstance(v, str):
                args["param_distributions"][k] = literal_eval(v)

    # if "mape" is selected as a metric, add it programmatically
    # as it is not built into sklearn
    if args["scoring"] and "mape" in args["scoring"]:
        args["scoring"]["mape"] = make_scorer(
            mean_absolute_percentage_error, greater_is_better=False
        )

    hp = utils.load_obj(params["tuner"]["class"])(**args)
    return hp


def sklearn_tune(
    params: dict, X: pd.DataFrame, y: np.ndarray, model: SklearnPipeline, **fit_kwargs
) -> List[Union[Any, Dict]]:
    """
    Generic tuning procedure for sklearn models.

    Args:
        params: dictionary of parameters
        X: training data X
        y: trainig data y
        model: sklearn pipeline with estimator and transformers
        cv: cross validation iterator
        **fit_kwargs: keyword args for `gs_cv.fit`
    Returns:
        Trained sklearn compatible model
        Hyper-parameter Tuning Search Results
    """
    gs_cv = get_hp_from_params(params, model)
    gs_cv.fit(X, y, **fit_kwargs)

    best_estimator = gs_cv.best_estimator_
    cv_results_df = pd.DataFrame(gs_cv.cv_results_)

    return best_estimator, cv_results_df


def xgb_tune(
    params: dict, X: pd.DataFrame, y: np.ndarray, model: SklearnPipeline
) -> List[Union[Any, Dict]]:
    """
    Tuning procedure for xgb estimators models. Under the hood,
    we use sklearn_tune but with a dedicated validation set for early stopping.

    Args:
        params: dictionary of parameters
        X: training data X
        y: training data y
        model: sklearn pipeline with estimator and transformers
        cv: cross validation iterator
    Returns:
        Trained sklearn compatible model
        Hyper-parameter Tuning Search Results
    """

    fit_kwargs = dict(estimator__verbose=False)

    return sklearn_tune(params, X, y.values, model, **fit_kwargs)


def grab_monotonicity(params: dict, td: TagDict) -> Dict[str, Any]:
    """
    Update model parameters to include any monotonity enforcement if provided
    in the tag dictionary. This allows the user to define the directionality
    of relationship that the model learns between a feature and the target

    Args:
        params: dictionary of parameters
        td: tag dictionary
    Returns:
        Dictionary of parameters, updated with monotonicity parameters if applicable

    """
    td_frame = td.to_frame()

    # If monotonicity column is in tagdict, grab the monotnicities to be enforced
    if "monotonicity" in td_frame.columns:
        if params["estimator"]["class"] in [
            "xgboost.XGBRegressor",
            "lightgbm.LGBMRegressor",
            "sklearn.ensemble.HistGradientBoostingRegressor",
        ]:
            feat_cols = params.get("features", "model_feature")
            feat_cols = td.select(feat_cols)
            monotonicities = (
                td_frame.loc[td_frame["tag"].isin(feat_cols), "monotonicity"]
                .fillna(0)
                .to_list()
            )

            # Define the name and format of the monotonicity parameter per algorithm
            param_format_dict = {
                "xgboost.XGBRegressor": dict(
                    name="monotone_constraints",
                    output="({0})".format(
                        ",".join(map(lambda x: str(int(x)), monotonicities))
                    ),
                ),
                "lightgbm.LGBMRegressor": dict(
                    name="monotone_constraints", output=monotonicities
                ),
                "sklearn.ensemble.HistGradientBoostingRegressor": dict(
                    name="monotonic_cst", output=monotonicities
                ),
            }

            # Add the monotonicities as a kwarg to the estimator parameters
            model_class = params["estimator"]["class"]
            params["estimator"]["kwargs"][
                param_format_dict[model_class]["name"]
            ] = param_format_dict[model_class]["output"]

        else:
            logger.warning(
                "Monotonicity not supported for selected estimator class. "
                "Please check documentation for supported estimators."
            )

    return params
