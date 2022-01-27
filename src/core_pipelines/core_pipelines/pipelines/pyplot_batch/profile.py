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
Nodes of the pyplot batch pipeline for the batch profiles.
"""
from ast import literal_eval
import logging
import math
from typing import Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

logger = logging.getLogger(__name__)


def get_series_limit(
    batches_time_series: pd.DataFrame,
    var_col: str,
    q: float = 0.05,
    time_unit: Optional[str] = None,
    start_time_scope: Optional[Union[int, float, np.datetime64]] = None,
    end_time_scope: Optional[Union[int, float, np.datetime64]] = None,
    extra_room: float = 0.1,
) -> Tuple[float, float]:
    """Computes the lineplot limits of a column of a given batch time series

    Args:
        batches_time_series: dataFrame with the time series of sensor for all batches
        var_col: Name of variable to compute the limits on
        q: The percentage of data to exclude when computing the limits
        time_unit: name of the column to compute the scope of the limit
        start_time_scope: start time to consider when computing the limit
        end_time_scope: end time to consider when computing the limit
        extra_room: after computing the limits, increase them by a margin

    Returns:
        The minimum and the maximum limit for the variable
    """

    # computing the scope if specified
    scope = pd.Series(
        np.ones(batches_time_series.shape[0]),
        index=batches_time_series.index,
        dtype=bool,
    )
    if time_unit is not None:
        if start_time_scope is not None:
            scope &= batches_time_series[time_unit] >= start_time_scope
        if end_time_scope is not None:
            scope &= batches_time_series[time_unit] <= end_time_scope

    # computing the limits using quantiles
    q_min = q / 2
    q_max = 1 - q / 2
    lim_min = batches_time_series.loc[scope, var_col].quantile(q=q_min)
    lim_max = batches_time_series.loc[scope, var_col].quantile(q=q_max)
    lim_range = lim_max - lim_min

    # adding a margin to the limits if specified
    if extra_room is not None:
        lim_min -= lim_range * extra_room
        lim_max += lim_range * extra_room

    return lim_min, lim_max


def get_all_series_limits(
    batches_time_series: pd.DataFrame,
    q: float = 0.05,
    time_unit: Optional[str] = None,
    start_time_scope: Optional[Union[int, float, np.datetime64]] = None,
    end_time_scope: Optional[Union[int, float, np.datetime64]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Returns a dictionary where the key is a column name of batches_time_series
    and the value is the min and max limits, for all numeric columns.
    Computed using the get_series_limit function.

    Args:
        batches_time_series: DataFrame with the time series of sensor for all batches
        q: The percentage of data to exclude when computing the limits.
        time_unit: name of the column to compute the scope of the limit
        start_time_scope: start time to consider when computing the limit
        end_time_scope: end time to consider when computing the limit

    Returns:
        Dictionary where the key is a column name of batches_time_series
        and the value is the min and max limits
    """

    plot_ranges = {}
    for col in batches_time_series.columns:

        # if the column has a numeric type
        # then get the limits and add it to the dictionary
        if (
            batches_time_series[col].dtype == np.float64
            or batches_time_series[col].dtype == np.int64
        ):
            plot_ranges[col] = get_series_limit(
                batches_time_series,
                col,
                time_unit=time_unit,
                q=q,
                start_time_scope=start_time_scope,
                end_time_scope=end_time_scope,
            )
    return plot_ranges


# pylint:disable=too-many-locals,too-many-arguments
# pylint:disable=,too-many-branches,too-many-statements
def plot_batch_profile(
    batch_time_series: pd.DataFrame,
    sensors_to_plot: Tuple[str, ...],
    time_unit: str,
    title: str,
    sensors_y_lim: Optional[Dict[str, Tuple[float, float]]] = None,
    start_time_scope: Optional[Union[int, float, np.datetime64]] = None,
    end_time_scope: Optional[Union[int, float, np.datetime64]] = None,
    nb_line_per_plot: int = 3,
    x_lim_max: Optional[Union[int, float, np.datetime64]] = None,
    vlines_x: Tuple[Union[int, float, np.datetime64], ...] = (),
    vlines_names: Tuple[str, ...] = (),
    figsize: Tuple[float, float] = (16, 9),
) -> Dict[str, plt.Figure]:
    """Create a figure with the traces of selected time series of a given batch

    Args:
        batch_time_series: the time series of all sensors from the batch
        sensors_to_plot: list of sensor names to plot
        time_unit: name of the column in batch_time_series to define the plotting scope
        title: the tile to give to the figure
        sensors_y_lim: dictionary of limits for time series plotting
        start_time_scope: start time to consider when plotting the traces
        end_time_scope: end time to consider when plotting the traces
        nb_line_per_plot: number of traces to plot per graph
        x_lim_max: absolute maximum range for the x axis
        vlines_x: list of timing to plot vertical lines on the profile
        vlines_names: list of names to assign to the vertical lines on the profile
        figsize: the size of the figure to create

    Returns:
        figure of profile
    """

    plt.close("all")

    # making sure every defined vertical line has a name
    assert len(vlines_names) == len(
        vlines_x
    ), "vlines_x and vlines_names must have the same lenght"

    # computing variables related to the presence of vertical lines to plot
    write_names = len(vlines_names) > 0
    write_names_int = int(write_names)

    # computing the number of rows of our figure depending on the number of sensors
    # to plot and the number of line plot per chart
    nb_sensors_to_plot = len(sensors_to_plot)
    nrows = math.ceil(nb_sensors_to_plot / nb_line_per_plot)

    # creating our subplots with as many rows as computed for the
    # lineplots and as an extra one if there are vertical line to plot
    fig, axes = plt.subplots(
        nrows=nrows + write_names_int, ncols=1, sharex="all", figsize=figsize
    )

    # if there is only one chart, then make it as a one element list for consistency
    if nrows == 1:
        axes = [axes]

    # fetching colors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_colors = len(colors)

    # fetching the scope of the batch to plot
    scope = pd.Series(
        np.ones(batch_time_series.shape[0]), index=batch_time_series.index, dtype=bool
    )
    if start_time_scope is not None:
        scope &= batch_time_series[time_unit] >= start_time_scope
    if end_time_scope is not None:
        scope &= batch_time_series[time_unit] <= end_time_scope

    # definition the x values, constant over all line plots
    x = batch_time_series.loc[scope, time_unit]

    # instantiate loop track
    i = 0

    # instantiate list of axe legends
    ax_legend = []

    # for each row of the subplots on which to plot sensor line plots
    for row_i in range(write_names_int, nrows + write_names_int):
        current_ax = axes[row_i]
        current_ax_legend = []

        # for each sensor to plot on current graph
        for sensor_i in range(nb_line_per_plot):

            sensor_name = sensors_to_plot[i]

            # plot the sensor trend
            current_ax.plot(
                x,
                batch_time_series.loc[scope, sensor_name],
                color=colors[i % n_colors],
                linestyle="-",
            )

            # set the axis tick in the same color as the lineplot
            current_ax.tick_params(axis="y", colors=colors[i % n_colors])

            # set the y limits
            if sensors_y_lim is not None:
                y_lim_min, y_lim_max = sensors_y_lim[sensor_name]
                current_ax.set_ylim(y_lim_min, y_lim_max)

            # create a legend and save it
            line_lgd = mlines.Line2D(
                [], [], color=colors[i % n_colors], linestyle="-", label=sensor_name
            )
            current_ax_legend.append(line_lgd)

            # if this is not the first sensor to plot on the current graph,
            # then set it on the right and offset to not overlap with previous ones
            if sensor_i >= 1:
                current_ax.spines["right"].set_position(
                    ("axes", 1 + 0.05 * (sensor_i - 1))
                )

            # disabling the grid if there are several sensors
            # on the same axe to avoid overlapping
            if nb_line_per_plot > 1:
                current_ax.grid(False)

            i += 1

            # as the last graph may have less sensors to plot on it
            # we must stop the loop once we reach the last one
            # pylint:disable=no-else-break
            if i == nb_sensors_to_plot:
                break
            # otherwise, create a new twin x axis on the current
            # graph for the next sensor line plot
            elif sensor_i < nb_line_per_plot - 1:
                current_ax = axes[row_i].twinx()

        # append the legends of previous chart to the general legend list
        ax_legend.append(current_ax_legend)

    # set the x axis limits as defined
    if x_lim_max is not None:
        axes[0].set_xlim(1, x_lim_max)

    # set the title as given
    axes[0].set_title(title, pad=(10))

    # remove horizontal space between charts
    fig.subplots_adjust(hspace=0)

    # setting the color of the vertical lines as the font color
    # to adapt to different plt style
    vline_color = plt.rcParams["text.color"]

    # for each graph, shrink horizontally to have space
    # of a legend box and add the legend
    for i in range(write_names_int, nrows + write_names_int):
        # Shrink current axis
        ax = axes[i]
        box = ax.get_position()
        ratio = 0.89
        ax.set_position(
            [box.width * (1 - ratio) + box.x0, box.y0, box.width * ratio, box.height]
        )
        ax.legend(
            handles=ax_legend[i - write_names_int],
            loc="center left",
            bbox_to_anchor=(-0.3, 0.5),
        )

        # plot vertical lines if any has been defined
        for vline_x in vlines_x:
            ax.axvline(vline_x, color=vline_color)

    # add the labels of defined vertical lines on a top graph, if any
    if write_names:
        # get the first chart, and remove its ticks
        ax = axes[0]
        ax.set_ylim(0, 10)
        ax.set_yticklabels([])
        ax.grid(False)
        ax.tick_params(
            axis="y",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False,
        )  # labels along the bottom edge are off

        # shrink it horizontally to be aligned with the bottom graphs
        box = ax.get_position()
        ratio = 0.89
        ax.set_position(
            [box.width * (1 - ratio) + box.x0, box.y0, box.width * ratio, box.height]
        )

        # draw a dotted line for each vertical line and add a label
        v_lines_x_len = len(vlines_x)
        for i in range(v_lines_x_len):
            vline_x = vlines_x[i]
            vline_name = vlines_names[i]
            ax.text(vline_x, 1, vline_name, rotation=30)
            ax.axvline(vline_x, color=vline_color, linestyle="--")

    return {"batch_profile": fig}


def plot_all_batches_profiles(
    batches_header: pd.DataFrame,
    batches_time_series: pd.DataFrame,
    sensors_to_plot: Tuple[str, ...],
    time_unit: str,
    batch_id_col: str,
    datetime_col: str,
    times_to_mark: Tuple[str, ...] = (),
    start_time_scope: Optional[Union[int, float, np.datetime64]] = None,
    end_time_scope: Optional[Union[int, float, np.datetime64]] = None,
    q: float = 0.05,
    nb_line_per_plot: int = 3,
    modular_title: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 9),
) -> Dict[str, Dict[str, plt.Figure]]:
    """Create a profile plot for all the batches fixing the axis limits

    Args:
        batches_header: the batches header data set with batch level information
        batches_time_series: dataFrame with the time series of sensors for all batches
        sensors_to_plot: list of sensor names to plot
        time_unit: column name in batch_time_series to use to define the plotting scope
        batch_id_col: column name from batches_time_series to use as batch id
        datetime_col: column name from batches_time_series to use as datetime
        times_to_mark: column names from batches_header that will be plotted vertically
        start_time_scope: start time to consider when plotting the traces
        end_time_scope: end time to consider when plotting the traces
        q: The percentage of data to exclude when computing the limits.
        nb_line_per_plot: number of traces to plot per graph
        modular_title: string to append to the title which will be evaluated
        figsize: the size of the figure for the profiles
    Returns:
        a dictionary of batches profiles

    """

    all_batch_profiles_fig = {}

    # select batche times series which are in the batches header
    batches_time_series = batches_time_series.loc[
        batches_time_series[batch_id_col].isin(batches_header.index)
    ]

    # get the limits for every column of batches time series,
    # will be useful to have the same limits across all
    # batches profiles
    plot_ranges = get_all_series_limits(
        batches_time_series,
        q=q,
        time_unit=time_unit,
        start_time_scope=start_time_scope,
        end_time_scope=end_time_scope,
    )

    # get the x axis limits which are the limits for the time unit
    lim_x = get_series_limit(
        batches_time_series,
        time_unit,
        time_unit=time_unit,
        q=0.00,
        start_time_scope=start_time_scope,
        end_time_scope=end_time_scope,
        extra_room=0,
    )

    grouped = batches_time_series.groupby([batch_id_col])

    for batch_id in batches_header.index:

        assert (
            batch_id in grouped.groups.keys()
        ), "batch id from batches header is not linked to a batch in times series"

        batch_time_series = grouped.get_group(batch_id)

        batch = batches_header.loc[batch_id]

        # set vlines_x by fetching its value from the batch header
        vlines_x = []
        if len(times_to_mark) > 0:
            vlines_x = batch[times_to_mark].values

        # create the title
        batch_start = batch_time_series[datetime_col].min()
        title = "Profile of Batch {}, started on {}".format(batch_id, batch_start)
        if modular_title is not None:
            title += "\n" + literal_eval(modular_title)

        logging.info("Plotting profile of batch {}".format(batch_id))

        batch_profile = plot_batch_profile(
            batch_time_series,
            sensors_to_plot,
            time_unit,
            title,
            sensors_y_lim=plot_ranges,
            start_time_scope=start_time_scope,
            end_time_scope=end_time_scope,
            nb_line_per_plot=nb_line_per_plot,
            x_lim_max=lim_x[1],
            vlines_x=vlines_x,
            vlines_names=times_to_mark,
            figsize=figsize,
        )

        all_batch_profiles_fig["batch_{}_profile".format(batch_id)] = batch_profile[
            "batch_profile"
        ]

    return {"all_batch_profiles": all_batch_profiles_fig}
