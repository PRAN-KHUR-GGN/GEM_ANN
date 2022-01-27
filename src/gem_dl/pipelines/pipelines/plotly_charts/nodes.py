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
Nodes of the plotly charts pipeline.
"""
import logging
from typing import Any, Dict
import pandas as pd


import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


logger = logging.getLogger(__name__)


def plotly_express_wrapper(data: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
    """Generates plotly graph object Figure based on the type and configs provided.

    See https://kedro.readthedocs.io/en/stable/_modules/kedro/extras/datasets/plotly/
    plotly_dataset.html#PlotlyDataSet

    Args:
        data: pandas dataframe to generate plotly Figure for
        params: plotly configurations for the desired plot

    Returns:
        A plotly graph_object figure representing the plotted data
    """
    fig_params = params.get("fig")
    plot = params.get("type")
    layout_params = params.get("layout", {})
    fig = getattr(px, plot)(data, **fig_params)
    fig.update_layout(layout_params)

    return fig


def plot_histogram(data: pd.DataFrame, params: Dict) -> Dict[str, go.Figure]:
    """
    Generates an interactive histogram plot.

    Args:
        data: the input dataset to plot
        params: parameters to be used

    Returns:
        Dictionary of "histogram_plot" -> Figure
    """
    internal_params = {
        "type": "histogram",
        "fig": {"x": ""},
        "layout": {"title": "Histogram", "template": "plotly"},
    }

    # update timeline params
    internal_params["fig"].update(params["fig"])
    internal_params["layout"].update(params.get("layout", {}))
    # Create figure and return
    fig = plotly_express_wrapper(data, internal_params)

    return {"histogram_plot": fig}


def plot_box(data: pd.DataFrame, params: Dict) -> Dict[str, go.Figure]:
    """
    Generates an interactive boxplot.

    Args:
        data: the input dataset to plot
        params: parameters to be used

    Returns:
        Dictionary of "box_plot" -> Figure
    """
    internal_params = {
        "type": "box",
        "fig": {"x": None, "y": ""},
        "layout": {"title": "Boxplot", "template": "plotly"},
    }

    # update timeline params
    internal_params["fig"].update(params["fig"])
    internal_params["layout"].update(params.get("layout", {}))
    # Create figure and return
    fig = plotly_express_wrapper(data, internal_params)

    return {"box_plot": fig}


def plot_scatter(data: pd.DataFrame, params: Dict) -> Dict[str, go.Figure]:
    """
    Generates a scatter plot of the given data according the params.

    Args:
        params: parameters to be used
        data: pandas dataframe to generate scatter for

    Returns:
        Dictionary of `scatter_plot` -> figure
    """
    internal_params = {
        "type": "scatter",
        "fig": {"x": "", "y": ""},
        "layout": {"title": "Scatter", "template": "plotly", "showlegend": True},
    }

    # update timeline params
    internal_params["fig"].update(params["fig"])
    internal_params["layout"].update(params.get("layout", {}))
    # Create figure and return
    fig = plotly_express_wrapper(data, internal_params)

    return {"scatter_plot": fig}


def plot_timeline(data: pd.DataFrame, params: Dict) -> Dict[str, go.Figure]:
    """
    Generates a timeline plot.

    Args:
        data: the pandas dataframe containing the timeline data
        params: the parameters to use

    Returns:
        Dictionary of `timeline_plot` -> go.Figure
    """
    internal_params = {
        "type": "line",
        "fig": {"x": "", "y": ""},
        "layout": {
            "title": "Timeline Plot",
            "xaxis_title": "Time",
            "template": "plotly",
            "showlegend": True,
        },
    }

    # update timeline params
    internal_params["fig"].update(params["fig"])
    internal_params["layout"].update(params["layout"])
    # Create figure and return
    fig = plotly_express_wrapper(data, internal_params)

    return {"timeline_plot": fig}


def plot_actual_vs_predicted(data: pd.DataFrame, params: Dict) -> Dict[str, go.Figure]:
    """
    Generates a plot of actual vs predicted values.

    Args:
        params: parameters to generate plot
        data: the pandas dataframe containing predictions and actual values

    Returns:
        Dictionary of `actual_vs_predicted_plot` -> figure
    """
    prediction_col = params.get("prediction_col", "prediction")
    target_col = params.get("target_col", "target")

    y_preds = data[prediction_col].rename("Predicted")
    y_true = data[target_col].rename("Actual")

    params["fig"] = {
        "x": "Predicted",
        "y": "Actual",
    }

    # If no layout key:
    # Add it with default
    # else use layout key
    params["layout"] = {**{"title": "Actual vs. Predicted"}, **params.get("layout", {})}
    pred_target: pd.DataFrame = pd.concat([y_true, y_preds], 1)
    fig = plot_scatter(pred_target, params)["scatter_plot"]

    return {"actual_vs_predicted_plot": fig}


def plot_single_feature_overview(
    data: pd.DataFrame, params: Dict
) -> Dict[str, go.Figure]:
    """
    Plots a collection of plots representing a feature "overview".
    This is includes a boxplot and histogram to understand the distribution of values,
    a scatterplot vs the target variable, and a time-series plot of the feature and
    target.

    Args:
        data: Dataframe holding data to plot
        params: Dictionary of params for the plotting functionality

    Returns:
        Dictionary of 'feature_overview' and go.Figure
    """
    feature_name = params.get("feature_name")
    target_name = params.get("target_name")
    timestamp_name = params.get("timestamp_name")
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{}, {}, {}], [{"colspan": 3, "secondary_y": True}, None, None]],
        subplot_titles=(
            "Boxplot",
            "Histogram",
            f"{feature_name} vs {target_name}",
            f"Historical development of {feature_name}",
        ),
        print_grid=False,
        column_widths=[0.15, 0.325, 0.325],
    )

    fig.add_trace(go.Box(y=data[feature_name], name=feature_name), row=1, col=1)
    fig.add_trace(go.Histogram(x=data[feature_name], name=feature_name), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=data[feature_name],
            y=data[target_name],
            name=f"{feature_name} vs {target_name}",
            mode="markers",
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=data[timestamp_name], y=data[target_name], name=target_name, mode="lines"
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data[timestamp_name],
            y=data[feature_name],
            name=feature_name,
            mode="lines",
        ),
        secondary_y=True,
        row=2,
        col=1,
    )

    # Unify the appearance of all of the plots
    default_layout_params = {
        "title": f"Overview - {feature_name}",
        "template": "plotly_dark",
        "showlegend": True,
        "height": 700,
    }
    default_layout_params.update(params.get("layout", {}))
    fig.update_layout(**default_layout_params)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig["layout"]["xaxis"].update(title=None, showticklabels=False)
    fig["layout"]["yaxis"].update(title=feature_name)

    fig["layout"]["yaxis2"].update(title="occurence")
    fig["layout"]["xaxis2"].update(title=feature_name)

    fig["layout"]["yaxis3"].update(title=target_name)
    fig["layout"]["xaxis3"].update(title=feature_name)
    fig["layout"]["yaxis4"].update(title=feature_name)
    fig["layout"]["yaxis5"].update(title=target_name)

    return {"feature_overview": fig}
