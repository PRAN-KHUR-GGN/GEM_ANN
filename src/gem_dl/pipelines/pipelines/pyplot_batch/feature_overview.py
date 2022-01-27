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
Nodes of the pyplot batch pipeline for the feature overview.
"""
from typing import Optional, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# pylint:disable=too-many-locals
def plot_feature_overview(
    feature_dataframe: pd.DataFrame,
    feature_column: str,
    target_column: str,
    hue_column: Optional[str],
    start_column: str,
) -> Dict[str, plt.Figure]:
    """Create exploratory charts for batch level features. Histogram,
    scatter and time behaviour plots will be generated

    Args:
        feature_dataframe: The dataframe with the feature to plot the overview of
        feature_column: the column from feature_dataframe to plot the overview of
        target_column: the column from feature_dataframe to set as target
        hue_column: the column from feature_dataframe to use a hue in the plots
        start_column: the column from feature_dataframe representing batch start
    Returns:
        A figure containing the overview of the selected feature in a dictionary
    """

    plt.close("all")

    fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=True)

    # Define grid
    grid = plt.GridSpec(5, 4, top=0.92, hspace=1.2, wspace=0.5, figure=fig)

    # Show hist
    ax_hist = fig.add_subplot(grid[0, :])
    sns.histplot(feature_dataframe[feature_column].dropna(), ax=ax_hist)
    ax_hist.set_title("Distribution")
    ax_hist.set_xlabel(feature_column)
    ax_hist.set_ylabel("Frequency")

    # Scatter plot VS target function
    ax_scatter = fig.add_subplot(grid[1:3, :], sharex=ax_hist)
    if hue_column is not None:
        # need this workaround to color scatter plot by hue
        sns.scatterplot(
            x=feature_column,
            y=target_column,
            hue=feature_dataframe[hue_column].tolist(),
            data=feature_dataframe,
            ax=ax_scatter,
        )
        # use regplot to plot the regression line for the whole points
        sns.regplot(
            x=feature_column,
            y=target_column,
            data=feature_dataframe,
            scatter=False,
            ax=ax_scatter,
        )
    else:
        sns.regplot(
            feature_column,
            target_column,
            data=feature_dataframe,
            scatter=True,
            ax=ax_scatter,
        )
    ax_scatter.set_title("Correlation")
    ax_scatter.set_xlabel(feature_column)
    ax_scatter.set_ylabel(target_column)

    # Show plot over time (rolling mean of 5 batches)
    ax_lineplot = fig.add_subplot(grid[3, :])
    ax_lineplot.plot(
        feature_dataframe[start_column],
        feature_dataframe[feature_column].rolling(window=5, center=False).mean(),
    )
    ax_lineplot.set_title("Rolling mean")
    ax_lineplot.set_ylabel(feature_column)
    ax_lineplot.set_xlabel(start_column)

    # Show descriptive
    plot = fig.add_subplot(grid[4, 0], frame_on=False)
    desc = feature_dataframe[feature_column].describe()
    index = desc.index
    desc = pd.Series(["%.2f" % d for d in desc])
    desc.index = index
    desc.name = "value"
    plot.xaxis.set_visible(False)
    plot.yaxis.set_visible(False)
    # setting the color of the table to the background
    bg_color = plt.rcParams["figure.facecolor"]
    rowColours = [bg_color] * desc.shape[0]
    colColours = [bg_color]
    cellColours = [[bg_color]] * desc.shape[0]
    pd.plotting.table(
        plot,
        desc,
        loc="center",
        colLabels=[],
        rowColours=rowColours,
        cellColours=cellColours,
        colColours=colColours,
    )

    # plot a boxplot per hue column value if not none
    if hue_column is not None:
        ax_boxplot = fig.add_subplot(grid[4, 1:])
        sns.boxplot(
            x=hue_column, y=feature_column, data=feature_dataframe, ax=ax_boxplot
        )
        ax_boxplot.set_title("Boxplot per {}".format(hue_column))
        ax_boxplot.set_ylabel(feature_column)

    fig.suptitle(feature_column)

    return {feature_column: fig}


def plot_all_features_overview(
    feature_dataframe: pd.DataFrame,
    target_column: str,
    hue_column: Optional[str],
    start_column: str,
    feature_columns: Tuple[str] = (),
) -> Dict[str, Dict[str, plt.Figure]]:
    """Create exploratory charts for batch level features.

     Histogram, scatter and time behaviour plots will be
     generated.

    Args:
        feature_dataframe: The dataframe with the feature to plot the overview of
        target_column: the column from feature_dataframe to set as target
        hue_column: the column from feature_dataframe to use a hue in the plots
        start_column: the column from feature_dataframe showing the batch start
        feature_columns:  the columns from feature_dataframe to plot the overview of
    Returns:
        One figure per feature, containing the overview of the selected feature
    """

    all_figs = dict()

    # fetching all numeric columns if no columns have been specified for feature_columns
    if len(feature_columns) == 0:
        feature_columns = feature_dataframe.select_dtypes(
            include=[np.number]
        ).columns.tolist()

    for feature_column in feature_columns:
        fig = plot_feature_overview(
            feature_dataframe, feature_column, target_column, hue_column, start_column
        )

        all_figs.update(fig)

    return {"all_features_overview": all_figs}
