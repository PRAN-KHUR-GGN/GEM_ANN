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
Nodes of the pyplot batch pipeline for the sensor trends.
"""

from typing import Optional, Dict, Tuple
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


# pylint:disable=too-many-locals
# pylint:disable=too-many-arguments
def plot_all_sensor_trends(
    batches_header: pd.DataFrame,
    batches_time_series: pd.DataFrame,
    sensors_to_plot: Tuple[str, ...],
    time_unit: str,
    batch_id_col: str,
    trend_type: str,
    hue: Optional[str] = None,
    alpha: Optional[float] = None,
    linewidth: float = 3,
    figsize: Tuple[float, float] = (16, 9),
    x_lim_quantile: float = 1,
) -> Dict[str, Dict[str, plt.Figure]]:
    """
    Compute sensor trends plot for batch analytics, can of type range or overlay

    Args:
        batches_header: The dataframe with information about all the batches
        batches_time_series: The dataframe containing the sensor dat
        sensors_to_plot: The list of sensors to plot
        time_unit: The column of batches_time_series to use on the x axis
        batch_id_col: The column in batches_time_series to use as batch id
        trend_type: The type of sensor trend to plot.
        hue: The column of batches_header to use as hue on the plot
        alpha: The transparency for the line in the plot
        linewidth: The line width in the plot
        figsize: The size of the figure
        x_lim_quantile: Allows to set a limit on the x axis using a quantile approach
    Returns:
        A dictionary of sensor trend plots
    """

    assert trend_type in ("overlay", "range"), "type must be 'overlay' or 'range'"

    plt.close("all")

    all_sensor_trends_fig = {}

    # merging batches header to batches time series to get batch level
    # info in batches times series and leverage
    # sns.lineplot hue parameter
    batches_time_series = batches_time_series.merge(
        batches_header, how="right", left_on=batch_id_col, right_index=True
    )

    # setting the right sns.lineplot parameters depending on the desired plot
    if trend_type == "overlay":
        estimator = None
        units = batch_id_col
    elif trend_type == "range":
        units = None
        estimator = "mean"

    if time_unit is None:
        # create a time step column which allows to align
        # the batches in time since beginning of batch
        time_unit = "time_step"
        i = 0
        while time_unit in batches_time_series.columns:
            time_unit = "time_step_{}".format(i)
            i += 1

        batches_time_series[time_unit] = (
            batches_time_series.groupby([batch_id_col]).cumcount() + 1
        )

    # setting the x axis limits
    x_lim_max = (
        batches_time_series.groupby([batch_id_col])[time_unit]
        .max()
        .quantile(q=x_lim_quantile)
    )
    x_lim_min = batches_time_series.groupby([batch_id_col])[time_unit].min().min()

    for sensor_to_plot in sensors_to_plot:

        logging.info("Plotting sensor {} for {}".format(trend_type, sensor_to_plot))

        fig, ax = plt.subplots(figsize=figsize)

        sns.lineplot(
            ax=ax,
            data=batches_time_series,
            x=time_unit,
            y=sensor_to_plot,
            units=units,
            hue=hue,
            estimator=estimator,
            alpha=alpha,
            linewidth=linewidth,
            ci=95,
        )

        ax.set_xlim((x_lim_min, x_lim_max))
        ax.set_title("Sensor {} for {}".format(trend_type, sensor_to_plot))
        all_sensor_trends_fig["sensor_trend_{}".format(sensor_to_plot)] = fig

    return {"all_sensor_trends": all_sensor_trends_fig}
