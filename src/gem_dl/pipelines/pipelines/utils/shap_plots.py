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

from contextlib import suppress
from typing import Dict, Tuple, List, Callable

import attr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optimus_core.core import utils
from sklearn.pipeline import Pipeline as SklearnPipeline

with suppress(ImportError):
    import shap


@attr.s
class ShapExplanation:
    """
    A simple class to hold the results of the SHAP stuff
    """

    shap_values = attr.ib()
    expectation = attr.ib()
    raw_features = attr.ib()


def plot_shap_summary(shap_result: ShapExplanation):
    """
    Creates a SHAP summary plot
    Args:
        shap_result: the shap explanation

    Returns:
        Plot figure
    """
    plt.close("all")

    shap_values = shap_result.shap_values
    X = shap_result.raw_features
    shap.summary_plot(
        shap_values.to_numpy(),
        X,
        feature_names=X.columns,
        plot_type="dot",
        plot_size=(16, 9),
        show=False,
    )

    plt.tight_layout()
    fig = plt.figure(1)

    return fig


def plot_abs_shap_summary(shap_result: ShapExplanation):
    """
    Creates an absolute SHAP plot
    Args:
        shap_result: the shap explanation

    Returns:
        Plot figure
    """
    plt.close("all")

    shap_values = shap_result.shap_values
    X = shap_result.raw_features
    feature_names = X.columns
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    corr_list = list()
    for f_name in feature_names:
        corr_coef = np.corrcoef(shap_df[f_name], X[f_name])[1][0]
        corr_list.append(corr_coef)

    corr_df = pd.concat(
        [pd.Series(feature_names), pd.Series(corr_list)], axis=1
    ).fillna(0)
    corr_df.columns = ["Variable", "Corr"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    corr_df["Sign"] = np.where(corr_df["Corr"] > 0, colors[3], colors[0])

    shap_abs = np.abs(shap_df)
    shap_abs = pd.DataFrame(shap_abs.mean()).reset_index()
    shap_abs.columns = ["Variable", "SHAP_abs"]
    shap_abs = shap_abs.merge(
        corr_df, left_on="Variable", right_on="Variable", how="inner"
    )
    shap_abs = shap_abs.sort_values(by="SHAP_abs", ascending=True)
    colorlist = shap_abs["Sign"]
    ax = shap_abs.plot.barh(
        x="Variable", y="SHAP_abs", color=colorlist, figsize=(16, 9), legend=False
    )
    ax.set_xlabel("SHAP Value")
    plt.legend(
        handles=[
            mpatches.Patch(color=colors[0], label="Negative Impact"),
            mpatches.Patch(color=colors[3], label="Positive Impact"),
        ]
    )

    plt.tight_layout()
    fig = plt.figure(1)

    # reverse sequence to have features in decreasing importance
    feat_cols = list(shap_abs["Variable"])[::-1]

    return feat_cols, fig


def plot_shap_single_dependence(col: str, shap_result: ShapExplanation):
    """
    Creates SHAP dependence plot for single feature
    Args:
        col: the feature to create plot for
        shap_result: a shap explanation

    Returns:
        Plot figure
    """
    plt.close("all")
    _, ax = plt.subplots(figsize=(16, 9))

    shap_values = shap_result.shap_values

    if isinstance(col, str):
        col_int = shap_values.columns.get_loc(col)
        if not isinstance(col_int, int):
            msg = "Duplicate column found in shap values? Col was: {}".format(col)
            raise ValueError(msg)
    else:
        col_int = col

    shap.dependence_plot(
        col_int,
        shap_values.values,
        shap_result.raw_features,
        color="C0",
        show=False,
        ax=ax,
    )
    plt.xlabel(str(col))
    plt.ylabel("SHAP value for {}".format(col))
    fig = plt.figure(1)

    return fig


def _get_shap_result(
    test_set_predictions: pd.DataFrame,
    model: SklearnPipeline,
    static_features: pd.DataFrame,
    shap_explainer: Callable,
    **shap_kwargs,
) -> Tuple[List[str], ShapExplanation]:

    feat_cols = model[:-1].transform(static_features.head()).columns.tolist()
    n_shap = min(200, len(test_set_predictions))

    train_data = static_features[feat_cols]
    test_data = test_set_predictions.sample(n=n_shap, random_state=0)[feat_cols]

    if isinstance(model, SklearnPipeline) and "estimator" in model.named_steps:
        model = model.named_steps["estimator"]
        # If the model is a KerasEstimator (and has the enable_prediction method to load
        # the keras model object itself, let's set the model to the underlying
        # keras model
        if hasattr(model, "enable_prediction"):
            model = model.enable_prediction().model

    if shap_explainer == shap.KernelExplainer:
        explainer = _create_kernel_explainer(model, train_data.values)
    else:
        explainer = shap_explainer(model, train_data.values)

    if shap_explainer == shap.DeepExplainer:
        shap_explanation = explainer.shap_values(
            test_data.values, check_additivity=False, **shap_kwargs
        )[0]
    else:
        shap_explanation = explainer.shap_values(
            test_data.values, check_additivity=False, **shap_kwargs
        )

    shap_frame = pd.DataFrame(
        shap_explanation, index=test_data.index, columns=test_data.columns
    )

    shap_result = ShapExplanation(shap_frame, explainer.expected_value, test_data)

    return feat_cols, shap_result


def _create_kernel_explainer(mdl, x):
    """
    Wraps a model and dataset to make a
    SHAP `KernelExplainer`.
    Args:
        mdl: The model, which has a `predict` function
        x: the dataset to explain

    Returns: a `shap.KernelExplainer`

    """
    try:
        mdl.predict
    except AttributeError:
        msg = "Model passed did not have predict function, model type: {}"
        raise TypeError(msg.format(type(mdl)))

    def inner_model_call(x_inner):
        return mdl.predict(x_inner)

    return shap.KernelExplainer(inner_model_call, x)


def generate_shap_figures(
    params: Dict,
    test_set_predictions: pd.DataFrame,
    model: SklearnPipeline,
    static_features: pd.DataFrame,
):
    """ Generate shap plots for your model

    Args:
        params (Dict): Dictionary of params
        test_set_predictions (pd.DataFrame): Dataframe of predictions from our model
        model (SklearnPipeline): Pipeline representing our model
        static_features (pd.DataFrame): Feature dataframe, likely model-input

    Returns:
        Dict[plt.Fig]: Dictionary of matplotlib figures, showing shap results.
    """
    shap_explainer = utils.load_obj(params["shap"]["explainer"])

    _, shap_result = _get_shap_result(
        test_set_predictions,
        model,
        static_features,
        shap_explainer,
        **params["shap"]["kwargs"],
    )

    feat_cols, shap_abs_summary = plot_abs_shap_summary(shap_result)
    # overwrite with user provided features, if available
    feat_cols = params["shap"].get("shap_features", feat_cols)

    return {
        "Shap Summary.png": plot_shap_summary(shap_result),
        "Shap Abs Summary.png": shap_abs_summary,
        **{
            f"Feature_{feature}.png": plot_shap_single_dependence(feature, shap_result)
            for feature in feat_cols
        },
    }
