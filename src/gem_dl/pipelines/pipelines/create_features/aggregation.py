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
Aggregation helper functions
"""
from typing import Callable, Union

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

# Create time delta object
one_day = pd.to_timedelta("1day")

# Subtract one day from the max and add one day to the minimum
# to avoid the maximum/minimum possible timestamp extending past
# their boundaries due to timezones. Localizing to and from MAX_TIME/MIN_TIME
# causes overflows since pandas caps the minimum/maximum timestamp
# at the equivalent of 2^-63 / 2^63 -1 unix time. Subtracting one day ensures we
# do not extend past the these values when converting through timezones
MAX_TIME = pd.Timestamp.max - one_day
MIN_TIME = pd.Timestamp.min + one_day


def agg_backwards(
    series: pd.Series, window: str, method: Union[str, Callable], closed="right"
) -> pd.Series:
    """
    Creates a backwards looking feature.

    Args:
        series: data series with datetime index
        window: pandas time frequency code
        method: aggregation method
        closed: closed argument for pandas rolling
    Returns:
        windowed feature
    """
    if not is_datetime64_any_dtype(series.index):
        raise ValueError(f"`series` must have a datetime index. Got `{series.index}`.")

    return series.rolling(window, closed=closed).agg(method)


def agg_backwards_categorical(
    series: pd.Series, window: str, method: Union[str, Callable], closed="right"
) -> pd.Series:
    """
    Creates a backwards looking feature for categorical features

    Args:
        series: categorical data series with datetime index
        window: pandas time frequency code
        method: aggregation method
        closed: closed argument for pandas rolling
    Returns:
        windowed feature
    """
    if not is_datetime64_any_dtype(series.index):
        raise ValueError(f"`series` must have a datetime index. Got `{series.index}`.")

    series = series.astype("category")
    return (
        series.cat.codes.rolling(window, closed=closed)
        .agg(method)
        .map(dict(enumerate(series.cat.categories)))
    )


def _reverse_time(index):
    # reverse time series
    max_time = MAX_TIME.tz_localize(index.tzinfo)
    min_time = MIN_TIME.tz_localize(index.tzinfo)
    diff = max_time - index
    rev = min_time + diff
    return rev


def _restore_time(rev_index):
    # restore original time series
    max_time = MAX_TIME.tz_localize(rev_index.tzinfo)
    min_time = MIN_TIME.tz_localize(rev_index.tzinfo)
    diff = rev_index - min_time
    orig = max_time - diff
    return orig


def agg_forwards(
    series: pd.Series, window: str, method: Union[str, Callable]
) -> pd.Series:
    """
    Creates a forwards looking feature.

    Args:
        series: data series with datetime index
        window: pandas time frequency code
        method: aggregation method
        closed: closed argument for pandas rolling
    Returns:
        windowed feature
    """
    if not is_datetime64_any_dtype(series.index):
        raise ValueError(f"`series` must have a datetime index. Got `{series.index}`.")

    # Forward looking features in pandas are significantly more complicated
    # than could possibly be justified. The recommended way to create a
    # forward-looking window is to create a backward-looking one, first
    # and then shift the results by the window size. However, this doesn't play
    # nicely with gaps in the time series. Instead, we use a hack here to
    # invert the time series, aggregate backwards, and then invert again.
    # This is reasonable fast since we are basically talking about adding
    # and subtracting integers. On the flip side, it doesn't work for dates
    # before 1970.
    # Please forgive me...
    idx_name = series.index.name
    rev = pd.Series(
        index=pd.Index(_reverse_time(series.index), name=idx_name), data=series.values
    )[::-1]
    agg = agg_backwards(rev, window, method, closed="left")
    return pd.Series(
        index=pd.Index(_restore_time(agg.index), name=idx_name), data=agg.values
    )[::-1]
