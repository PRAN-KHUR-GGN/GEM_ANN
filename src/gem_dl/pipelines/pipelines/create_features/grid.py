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
Grid creation nodes.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _create_grid_from_series(
    time_series: pd.Series,
    freq: str,
    offset_start: str = None,
    offset_end: str = None,
    timezone: str = None,
) -> pd.DatetimeIndex:
    """
    Create a time grid from the first to the last timestamp in a time series
    with optional offsets at both ends. See
    https://pandas.pydata.org/pandas-docs/version/0.25.0/
    user_guide/timeseries.html#timeseries-offset-aliases
    for valid frequency strings.
    Resulting time stamps are rounded in order to get merge-able grids
    for different data sources.
    Args:
        time_series: series of timestamps
        freq: pandas frequency string
        offset_start: time to add to first timestamp
        offset_end: time to subtract from last timestamp
        timezone: timezone information
    Returns:
        time grid
    """
    time_series = pd.to_datetime(time_series)
    lhs, rhs = time_series.agg(["min", "max"])
    if offset_start:
        lhs += pd.to_timedelta(offset_start)
    if offset_end:
        rhs -= pd.to_timedelta(offset_end)

    return pd.date_range(lhs.ceil(freq), rhs.floor(freq), freq=freq, tz=timezone)


def create_time_grid(params: dict, *dfs: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Creates a time grid from a data frame.
    Args:
        params: dictionary of parameters
        dfs: any number of dataframes with time series column
    """
    freq = params["grid"]["frequency"]
    offset_start = params["grid"].get("offset_start")
    offset_end = params["grid"].get("offset_end")
    timezone = params["pipeline_timezone"]

    all_timestamps = pd.concat([df["timestamp"] for df in dfs], ignore_index=True)

    return _create_grid_from_series(
        all_timestamps, freq, offset_start, offset_end, timezone
    )
