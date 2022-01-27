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
Nodes of the timeseries data stitching pipeline.
"""
import logging

import numpy as np
import pandas as pd


from optimus_core.core.tag_management import TagDict
from ..utils import get_valid_agg_method, resample_dataframe

logger = logging.getLogger(__name__)

MERGE_STRATEGIES = ["match", "resample", "merge_asof"]


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
    master_timestamp_col = params["master_timestamp_col"]
    timezone = params["pipeline_timezone"]

    all_timestamps = pd.concat(
        [df[master_timestamp_col] for df in dfs], ignore_index=True
    )

    return _create_grid_from_series(
        all_timestamps, freq, offset_start, offset_end, timezone
    )


def merge_to_grid(  # pylint: disable=too-many-locals
    grid: pd.DatetimeIndex, params: dict, td: TagDict, *to_merge: pd.DataFrame
) -> pd.DataFrame:
    """
    Left-merges any number of dataframes to the grid.
    Merge strategy defined by parameters for each dataframe
    to be merged.

    Ensures that the size of the resulting df remains unchanged.
    Args:
        grid: time points at which to aggregate
        params: dictionary of parameters
        td: data dictionary
        to_merge: any number of sources and source parameters to merge in
    Returns:
        merged df
    """
    n_rows = len(grid)

    master_timestamp_col = params["master_timestamp_col"]
    merged = pd.DataFrame({master_timestamp_col: grid})
    grid_freq = params["grid"].get("frequency")
    source_params = params.get("sources")  # source-params is List of dicts
    merge_strategies = [source.get("merge_strategy") for source in source_params]

    # Raise if a merge strategy isn't implemented yet
    if not all(
        [merge_strategy in MERGE_STRATEGIES for merge_strategy in merge_strategies]
    ):
        raise NotImplementedError(
            f"You've supplied {np.unique(merge_strategies)}, which includes some "
            "currently un-implemented strategies. "
            f"Please select from strategies {MERGE_STRATEGIES}"
        )

    # Raise if a dataset doesn't have params, or params doesn't have a dataset.
    if len(source_params) != len(to_merge):
        raise ValueError(
            f"You've specified params for {len(source_params)} datasets, "
            f"supplied {len(to_merge)} datasets to join/merge"
        )
    for idx, df in enumerate(to_merge):
        df[master_timestamp_col] = pd.to_datetime(df[master_timestamp_col])
        merge_strategy = source_params[idx].get("merge_strategy")

        if merge_strategy == "match":
            merged = pd.merge(
                merged,
                df,
                how="left",
                left_on=master_timestamp_col,
                right_on=master_timestamp_col,
            )
        elif merge_strategy == "resample":
            merged = _merge_resample(df, grid_freq, master_timestamp_col, merged, td)
        elif merge_strategy == "merge_asof":
            merged = _merge_asof(df, grid_freq, master_timestamp_col, merged)
        if len(merged) != n_rows:
            raise RuntimeError(
                f"Merging dataframe {idx+1} led to a change in the number of rows "
                f"from {n_rows} to {len(merged)}. Please check for duplicate "
                "timestamps."
            )

    return merged


def _merge_asof(df, grid_freq, master_timestamp_col, merged):
    cols = df.drop(master_timestamp_col, axis=1).columns
    tolerance = pd.Timedelta(grid_freq)
    for col in cols:
        subset = df[[master_timestamp_col, col]].dropna()
        merged = pd.merge_asof(
            merged,
            subset,
            left_on=master_timestamp_col,
            right_on=master_timestamp_col,
            allow_exact_matches=True,
            tolerance=tolerance,
            direction="nearest",
        )
    return merged


def _merge_resample(df, grid_freq, master_timestamp_col, merged, td):
    resampling_methods = dict()
    data_cols = df.drop(master_timestamp_col, axis=1).columns.values
    for col in data_cols:
        method = get_valid_agg_method(col, td, errors="raise", default_method=None)
        resampling_methods[col] = method
    df_res = resample_dataframe(df, master_timestamp_col, grid_freq, resampling_methods)
    merged = pd.merge(
        merged,
        df_res,
        how="left",
        left_on=master_timestamp_col,
        right_on=master_timestamp_col,
    )
    return merged
