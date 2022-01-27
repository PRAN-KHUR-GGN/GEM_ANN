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
Create target and control features
"""
import logging
from multiprocessing import Pool

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from tqdm import tqdm

from optimus_core.core.tag_management import TagDict

from .aggregation import agg_backwards, agg_forwards, agg_backwards_categorical

logger = logging.getLogger(__name__)


def _single_value_mode(x: pd.Series):
    return x.mode()[0]


KNOWN_AGG_METHODS = {"mode": _single_value_mode}


def _add_grid(df: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Adds empty rows to the dataframe at the timepoints specified
    in the grid. This is necessary, because pandas does not have a way
    to do rolling operations with step size.
    """
    assert is_datetime64_any_dtype(df.index)
    assert is_datetime64_any_dtype(grid)

    # don't add indices that already exist
    missing_indices = grid.difference(df.index)

    dummy_df = pd.DataFrame(index=missing_indices, columns=df.columns)
    dummy_df.index.name = df.index.name

    return pd.concat([df, dummy_df]).sort_index()


def _reduce_to_grid(df: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Reduces a dataframe to only those indices in `grid`.
    Drops duplicates.
    """
    grid_only = df.loc[grid, :]
    deduped = grid_only[~grid_only.index.duplicated(keep="first")]
    deduped.index.name = df.index.name
    return deduped


def _agg_dict(kwargs):
    func = kwargs.pop("func")
    col = kwargs.pop("col")
    return col, func(**kwargs)


def create_train_features(  # pylint:disable=too-many-locals
    params: dict, td: TagDict, data: pd.DataFrame, grid: pd.DatetimeIndex
):
    """
    Creates a forward-looking modelling target with backward looking controls.

    Args:
        params: dictionary of parameters
        td: tag dictionary
        data: input data
        grid: time points at which to aggregate
    Returns:
        dataframe with aggregated features at grid intervals
    """
    return _aggregate_tags(params, td, data, grid, False)


def create_opt_features(  # pylint:disable=too-many-locals
    params: dict, td: TagDict, data: pd.DataFrame, grid: pd.DatetimeIndex
):
    """
    Creates a forward-looking modelling target with forward looking controls.

    Args:
        params: dictionary of parameters
        td: tag dictionary
        data: input data
        grid: time points at which to aggregate
    Returns:
        dataframe with aggregated features at grid intervals
    """
    return _aggregate_tags(params, td, data, grid, True)


def _aggregate_tags(  # pylint:disable=too-many-locals
    params: dict,
    td: TagDict,
    data: pd.DataFrame,
    grid: pd.DatetimeIndex,
    backwards_controls=False,
) -> pd.DataFrame:
    """
    Creates the forward-looking modelling target.

    Args:
        params: dictionary of parameters
        td: tag dictionary
        data: input data
        grid: time points at which to aggregate
        backwards_controls: if True, make controls and on-off
        tags bw instead of fw looking
    Returns:
        dataframe with aggregated features at grid intervals
    """
    n_jobs = params["n_jobs"]

    # we only process columns in the tag dict
    td_cols = [c for c in data.columns if c in td]

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    with_grid = _add_grid(data.set_index("timestamp")[td_cols], grid)

    # prepare kwarg dicts for parallel aggregation
    agg_dicts = []

    for col in td_cols:
        col_info = td[col]

        window = col_info["agg_window_length"]
        method = col_info["agg_method"]

        if not window or not method:
            raise RuntimeError(
                f"Could not find aggregation window and/or method "
                f"for `{col}` in the tag dictionary."
            )

        # use custom methods where available
        method = KNOWN_AGG_METHODS.get(method, method)

        if col_info["model_target"]:
            # targets are always calculated forwards
            func = agg_forwards
        elif col_info["tag_type"] in ["control", "on_off"]:
            # here, it depends: when optimizing, we look backwards to
            # estimate the current state of our controls. Otherwise
            # we look forwards to create training data
            if backwards_controls:
                func = agg_backwards
            else:
                func = agg_forwards
        elif (data[col].dtype.name in ["object", "category"]) & (
            col_info["agg_method"] == "mode"
        ):
            func = agg_backwards_categorical
        else:
            # default is backwards
            func = agg_backwards

        agg_dicts.append(
            dict(
                col=col, series=with_grid[col], func=func, window=window, method=method
            )
        )

    # we use imap (lazy pool.map) here to make tqdm work
    with Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(_agg_dict, agg_dicts), total=len(td_cols)))
        pool.close()
        pool.join()

    combined = pd.DataFrame(dict(results))
    combined.index.name = with_grid.index.name

    return _reduce_to_grid(combined, grid).reset_index()


def merge_to_grid(grid: pd.DatetimeIndex, *to_merge: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merges any number of dataframes to the grid.
    Ensures that the size of the resulting df remains unchanged.
    Args:
        grid: time points at which to aggregate
        to_merge: any number of dataframes to merge in
    Returns:
        merged df
    """
    n_rows = len(grid)

    merged = pd.DataFrame({"timestamp": grid})

    for i, df in enumerate(to_merge):
        merged = pd.merge(
            merged, df, how="left", left_on="timestamp", right_on="timestamp"
        )
        if len(merged) != n_rows:
            raise RuntimeError(
                (
                    "Merging dataframe {} led to a change in the number of rows "
                    "from {} to {}. Please check for duplicate timestamps."
                ).format(i + 1, n_rows, len(merged))
            )

    return merged
