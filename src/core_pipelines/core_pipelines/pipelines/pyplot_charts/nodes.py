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
Nodes of the pyplot charts pipeline.
"""
import logging
from typing import Dict, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from dateutil.parser import ParserError
from matplotlib.figure import Figure
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox
from sklearn.base import BaseEstimator
from sklearn.inspection import plot_partial_dependence

logger = logging.getLogger(__name__)


def plot_box(data: pd.DataFrame, params: Dict) -> Dict[str, Figure]:
    """
    Generates a box plot that is optionally binned as described in params
    Args:
        data: the input dataset to plot
        params: parameters to be used

    Returns:
        Dictionary of "box_plot" -> Figure
    """

    plt.close("all")
    by_column = params.get("by", {}).get("column", None)
    by_bins = params.get("by", {}).get("bins", None)
    by_quantiles = params.get("by", {}).get("quantiles", None)

    if by_bins:

        if isinstance(by_bins, list) and all(
            isinstance(elem, str) for elems in by_bins for elem in elems
        ):
            # if bins are lists of strings, then group by the given buckets
            bins = {v: i for i, l in enumerate(by_bins) for v in l}
            data[by_column] = data[by_column].map(bins)

        elif isinstance(by_bins, list) and all(len(elems) == 2 for elems in by_bins):
            # if bins are tuples of lower / upper bounds, treat as intervals
            bins = pd.cut(data[by_column], bins=pd.IntervalIndex.from_tuples(by_bins))
            data[by_column] = bins

        else:
            # otherwise assume argument to cut
            bins = pd.cut(data[by_column], bins=by_bins)
            data[by_column] = bins

    elif by_quantiles:
        # if by quantiles then qcut data
        bins = pd.qcut(data[by_column], q=by_quantiles)
        data[by_column] = bins

    fig, ax = plt.subplots(figsize=params.get("figsize", (16, 9)))
    data.boxplot(column=params["column"], by=by_column, ax=ax)

    if by_bins and isinstance(by_bins, list):
        ax.set_xticklabels(by_bins)

    if "x_label" in params:
        ax.set_xlabel(params["x_label"])
    ax.set_ylabel(params.get("y_label", params["column"]))
    plt.suptitle(params.get("title", ""))
    ax.set_title(params.get("subtitle", ""))

    return {"box_plot": fig}


def plot_correlation(data: pd.DataFrame, params: Dict) -> Dict[str, Figure]:
    """
    Plots correlation matrix for the features with each-other in form of a heatmap.

    Args:
        data: pandas dataframe for which the analysis has to be performed
        params: paramters to be used

    Returns:
        Dictionary of `correlation_plot` -> figure

    """
    plt.close("all")

    plot_data = generate_correlation_matrix(data, params)["corr_df"]
    # Mask for the upper half triangle in the heatmap
    mask = np.triu(plot_data) if params["mask"] else None

    # Plotting the heatmap plot
    fig = plt.figure(figsize=params.get("figsize", (16, 9)))
    sns.heatmap(
        plot_data, annot=True, mask=mask, cmap="vlag", center=0, square=True,
    )

    fig.suptitle(params.get("title", "Correlation Plot"))
    fig.tight_layout()

    return {"correlation_plot": fig}


def generate_correlation_matrix(
    plot_data: pd.DataFrame, params: Dict
) -> Dict[str, pd.DataFrame]:
    """
    Calculate the correlation matrix for the provided dataset
    Args:
        plot_data: dataset to calculate the correlation for
        params: parameters to be used
    Returns:
        correlation table and mask array
    """

    corr_df = plot_data.corr(method=params.get("method", "pearson"))

    if params.get("clustered", True):
        g = sns.clustermap(corr_df)
        # rearrange DataFrame into a clustered version
        icol = g.dendrogram_col.reordered_ind
        irow = g.dendrogram_row.reordered_ind
        corr_df = corr_df.iloc[irow, icol]

    return {"corr_df": corr_df}


def plot_scatter(data: pd.DataFrame, params: Dict) -> Dict[str, plt.figure]:
    """
    Generates scatter plot of the given data according the params

    Args:
        params: parameters to be used
        data: pandas dataframe to generate scatter for

    Returns:
        Dictionary of `scatter_plot` -> figure
    """
    plt.close("all")

    fig, ax = plt.subplots(figsize=params.get("figsize", (16, 9)))
    size = data[params["size_col"]] if "size_col" in params else None

    if isinstance(params["y_col"], str):
        params["y_col"] = [params["y_col"]]
    elif "color_col" in params:
        raise ValueError("color_col is not supported with multiple y_col")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, y_col in enumerate(params["y_col"]):
        data.plot.scatter(
            x=params["x_col"],
            y=y_col,
            label=y_col,
            s=size,
            c=params.get("color_col", colors[i % len(colors)]),
            title=params.get("title", "Scatter Plot"),
            ax=ax,
        )

    return {"scatter_plot": fig}


def plot_density(params: Dict, data: pd.DataFrame) -> Dict[str, plt.figure]:
    """
    Generate a density plot from the given data and parameters

    Args:
        params: the parameters to use
        data: the pandas dataframe to generate density plot from

    Returns:
        Dictionary of `density_plot` -> figure
    """
    plt.close("all")

    fig, _ = plt.subplots(figsize=params.get("figsize", (16, 9)))

    for col in params["columns"]:
        sns.distplot(
            data[[col]],
            hist=params.get("hist", False),
            hist_kws=params.get("hist_kws", {}),
            kde=params.get("kde", True),
            kde_kws=params.get("kde_kws", {}),
            label=col,
        )

    plt.title(params.get("title", "Density Plot"))
    plt.xlabel(params.get("xlabel", ""))
    plt.ylabel(params.get("ylabel", "Density"))

    return {"density_plot": fig}


def plot_partial_dependency(
    params: Dict, data: pd.DataFrame, model: BaseEstimator
) -> Dict[str, plt.figure]:
    """
    Generate a partial dependency plot
    Args:
        params: parameters to use
        data: the dataframe to generate the pdp
        model: base estimator to use for pdp
    Returns:
        Dictionary of `partial_dependency_plot` -> figure
    """
    plt.close("all")

    fig, ax = plt.subplots(figsize=params.get("figsize", (16, 9)))

    if isinstance(model, sklearn.pipeline.Pipeline):
        # the 'plot_partial_dependency' method has a bug in it where
        # it expects every step an estimator objects.
        # Where this is not true, it tries to check that it
        # has been fit, and thus is useable. To "trick" the method
        # past this, we can add an attribute ending _ to each step,
        # allowing the sklearn `check_is_fit` method to pass.
        # They promise future support for pipelines!
        for _, step in model.steps:
            step.dummy_fit_ = "dummy"

    plot_partial_dependence(
        estimator=model, X=data, features=params.get("columns", data.columns), ax=ax
    )

    fig.suptitle(params.get("title", "Partial Dependency Plot"))
    for axis in [axis for axis in fig.get_axes() if "ylabel" in params]:
        axis.set_ylabel(params["ylabel"])

    return {"partial_dependency_plot": fig}


def plot_bar(params, data):
    """
    Generate a bar plots from the given data and parameters
    Args:
        params: the parameters to use
        data: the data to plot

    Returns:
        Dictionary of `bar_plot` -> figure
    """
    plt.close("all")

    fig, ax = plt.subplots(figsize=params.get("figsize", (16, 9)))

    if "sort_by" in params:
        data = data.sort_values(by=params["sort_by"], ascending=True)

    data.plot.barh(
        x=params.get("x", None),
        y=params.get("y", None),
        ax=ax,
        title=params.get("title", "Bar Plot"),
    )

    plt.ylabel(params.get("y_label", params.get("x", None)))

    if params.get("annotate", False):
        for p in ax.patches:
            ax.annotate(
                f"{p.get_width():.2f}", (p.get_width() * 1.005, p.get_y() * 1.005)
            )
    plt.tight_layout()

    return {"bar_plot": fig}


def plot_actual_vs_predicted(params: Dict, data) -> Dict[str, plt.figure]:
    """
    Generate plot of actual vs predicted values

    Args:
        params: parameters to generate plot
        data: the pandas dataframe containing predictions and actual values

    Returns:
        Dictionary of `actual_vs_predicted_plot` -> figure
    """
    plt.close("all")

    prediction_col = params.get("prediction_col", "prediction")
    target_col = params.get("target_col", "target")

    y_preds = data[prediction_col].rename("Predicted")
    y_true = data[target_col].rename("Actual")

    pred_target: pd.DataFrame = pd.concat([y_true, y_preds], 1)

    scatter_ax = pred_target.plot.scatter(
        x="Actual",
        y="Predicted",
        figsize=params.get("figsize", (16, 9)),
        title=params.get("title", "Actual vs. Predicted"),
    )

    xmin_lim, xmax_lim = scatter_ax.get_xlim()
    scatter_ax.set_ylim(xmin_lim, xmax_lim)

    plt.plot(scatter_ax.get_xlim(), scatter_ax.get_ylim(), linestyle="--")

    fig = plt.figure(1)

    return {"actual_vs_predicted_plot": fig}


def plot_actual_vs_residual(params: Dict, data: pd.DataFrame) -> Dict[str, plt.figure]:
    """
    Generate a plot of actual vs residual values

    Args:
        params: the parameters to use
        data: the pandas dataframe containing the actual and residual values

    Returns:
        Dictoniary of `actual_vs_residual_plot` -> figure
    """
    plt.close("all")

    prediction_col = params.get("prediction_col", "prediction")
    target_col = params.get("target_col", "target")

    y_preds = data[prediction_col].rename("Predicted")
    y_true = data[target_col].rename("Actual")

    residuals: pd.Series = (y_true - y_preds).rename("Residual")
    res_df: pd.DataFrame = pd.concat([y_true, residuals], 1)
    axis_val = max(abs(residuals.min()), abs(residuals.max())) * 1.05
    residuals_ax = res_df.plot.scatter(
        "Actual",
        "Residual",
        figsize=params.get("figsize", (16, 9)),
        title=params.get("title", "Actual vs. Residuals"),
    )
    residuals_ax.set_ylim(-1 * axis_val, axis_val)
    plt.axhline(linestyle="-")

    fig = plt.figure(1)

    return {"actual_vs_residual_plot": fig}


def plot_timeline(params: Dict, data: pd.DataFrame) -> Dict[str, plt.figure]:
    """
    Generate a timeline plot

    Args:
        params: the parameters to use
        data: the pandas dataframe containing the timeline data

    Returns:
        Dictionary of `timepline_plot` -> figure
    """
    plt.close("all")

    cols = params["columns"]
    timestamp_col = params.get("timestamp_col", "timestamp")

    data = data[[timestamp_col, *cols]].copy()

    try:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    except ParserError:
        logging.warning(f"{timestamp_col} is not a timestamp.")

    data = data.set_index(timestamp_col)

    fig, ax = plt.subplots(figsize=params.get("figsize", (16, 9)))
    data.head(params.get("n", 1000)).plot(ax=ax)

    plt.title(params.get("title", "Timeline"))
    plt.ylabel(params.get("ylabel", ""))

    fig = plt.figure(1)

    return {"timeline_plot": fig}


def plot_string(params: Dict, string_data: Any) -> Dict[str, plt.figure]:
    """
    Generate a text plot from any data that can be converted to string

    Args:
        params: the parameters to use
        string_data: the data to print

    Returns:
        Dictionary of `string_plot` -> figure
    """
    plt.close("all")

    fig = plt.figure(figsize=params.get("figsize", (16, 9)))
    _, ax = plt.subplots(1, 1)

    text = [
        TextArea(
            f"{params.get('title', '')}", textprops=dict(fontsize=24, fontweight="bold")
        ),
        *[
            TextArea(s, textprops=dict(fontsize=14, fontweight="normal", wrap=True))
            for s in str(string_data).split("\\n")
        ],
    ]

    texts_vbox = VPacker(children=text, pad=30, sep=30)
    ann = AnnotationBbox(
        texts_vbox,
        (0.0, 0.5),
        xycoords=ax.transAxes,
        box_alignment=(0, 0),
        bboxprops=dict(color="black", boxstyle="round", alpha=0),
    )
    ann.set_figure(fig)
    fig.artists.append(ann)

    return {"string_plot": fig}


def plot_pandas_df(params: Dict, data: pd.DataFrame) -> Dict[str, plt.figure]:
    """
    Generate a text output of a pandas dataframe

    Args:
        params: the parameters to use
        data: the dataframe to show

    Returns:
        Dictinary of `pandas_df_plot` -> figure
    """
    plt.close("all")

    fig = plt.figure(figsize=params.get("figsize", (16, 9)))
    _, ax = plt.subplots(1, 1)

    text = [
        TextArea(
            f"{params.get('title', 'Dataframe')}",
            textprops=dict(fontsize=24, fontweight="bold"),
        ),
        TextArea(
            str(data),
            textprops=dict(fontsize=14, fontweight="normal", family="monospace"),
        ),
    ]

    texts_vbox = VPacker(children=text, pad=10, sep=10)
    ann = AnnotationBbox(
        texts_vbox,
        (0, 0.5),
        xycoords=ax.transAxes,
        box_alignment=(0, 0),
        bboxprops=dict(color="black", facecolor="#1F78B4", boxstyle="round", alpha=0),
    )
    ann.set_figure(fig)
    fig.artists.append(ann)

    return {"pandas_df_plot": fig}


def plot_timeline_column(params: Dict, data: pd.DataFrame) -> Dict[str, plt.figure]:
    """
    Generate a timeline column chart from the given data

    Args:
        params: the parameters to use
        data: the dataframe to show

    Returns:
        Dictionary of "timeline_column_plot" -> figure
    """
    plt.close("all")

    col = params["column"]
    timestamp_col = params.get("timestamp_col", "timestamp")

    # create plot
    fig, ax = plt.subplots(
        figsize=params.get("figsize", (16, 9)), constrained_layout=True
    )

    # convert timestamp (if possible)
    data = data[[timestamp_col, col]].copy()
    try:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        ticks = data[timestamp_col].dt.strftime("%Y-%m-%d %H-%M-%S")
    except ParserError:
        ticks = data[timestamp_col]
        logging.warning(f"{timestamp_col} is not a timestamp.")

    # grab data to plot
    arr = [data[col].values]
    norm = mcolors.Normalize(vmin=np.min(arr), vmax=np.max(arr))

    # plot image and color bar
    im = ax.pcolormesh(arr, rasterized=True, norm=norm)
    if params.get("legend", True):
        fig.colorbar(im, ax=ax, shrink=0.6)

    # configure chart
    ax.set_xticklabels(ticks)
    ax.axes.get_yaxis().set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.title(params.get("title", "Timeline"))

    fig = plt.figure(1)

    return {"timeline_column_plot": fig}
