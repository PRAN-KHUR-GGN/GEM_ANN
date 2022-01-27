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
Resampling creation nodes.
"""
import logging

import pandas as pd
import numpy as np
from optimus_core.core.tag_management import TagDict

logger = logging.getLogger(__name__)


def _single_value_mode(x: pd.Series):
    return x.mode()[0]


KNOWN_AGG_METHODS = {"mode": _single_value_mode}
ALLOWED_AGG_METHODS = ["mean", "sum", "median", "mode"]


def resample_data(params: dict, params_ts, data: pd.DataFrame, td: TagDict):
    """Resample data to resample frequency.

    Resample according to agg_method defined in data dictionary.
    Args:
        params: dictionary with resampling parameters
        params_ts: dictionary with master timestamp column parameters
        data: input data
        td: data dictionary
    Returns:
        data_resampled: resampled output data
    """

    resample_freq = params.get("resample_freq")
    errors = params.get("errors")
    default_method = params.get("default_method")

    ts_col = params_ts.get("master_timestamp_col")

    resampling_methods = dict()
    data_cols = data.drop(ts_col, axis=1).columns.values
    for col in data_cols:
        method = get_valid_agg_method(col, td, errors, default_method)
        resampling_methods[col] = method

    return resample_dataframe(data, ts_col, resample_freq, resampling_methods)


def get_valid_agg_method(tag: str, td: TagDict, errors: str, default_method: str):
    """Select valid aggregation method for a tag.

    Selects the aggregation method for a tag from the
    data dictionary. If not defined, raise error or default
    to a default aggregation method.

    Args:
        tag: string of the tag
        td: data dictionary
        errors: str {'raise', 'coerce'}, raise errors if tag has no agg_method
            defined in data dictionary or coerce to default
    Returns:
        data_resampled: resampled output data
    """

    method = td[tag]["agg_method"]
    if errors == "raise" and method is np.nan:
        raise ValueError(f"No aggregation method defined for column {tag}")
    if errors == "coerce" and method is np.nan:
        method = default_method

    if method not in ALLOWED_AGG_METHODS:
        raise ValueError(
            f"Aggregation method {method} not allowed. "
            f"Please select from {ALLOWED_AGG_METHODS}"
        )

    method = KNOWN_AGG_METHODS.get(method, method)

    return method


def resample_dataframe(
    data: pd.DataFrame, ts_col: str, resample_freq: str, resampling_method: dict
):
    """Resample dataframe columns according to resampling methods.

    Applies resampling_method to each column separately to
    resample to specified resample_freq.

    Args:
        data: input dataframe,
        ts_col: name of the timestamp column,
        resample_freq: timedelta string of frequency to resample to,
        resampling_method: dictionary mapping resampling function to columns
    Returns:
        data_resampled: resampled output data
    """
    data[ts_col] = pd.to_datetime(data[ts_col])
    data = data.set_index(ts_col, drop=True)
    data_resampled = data.resample(rule=resample_freq).apply(resampling_method)
    data_resampled = data_resampled.reset_index()

    return data_resampled
