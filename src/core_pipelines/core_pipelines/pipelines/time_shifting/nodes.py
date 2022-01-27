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
Time shifting node.
"""
import logging
from itertools import groupby

import pandas as pd

logger = logging.getLogger(__name__)


def _check_time_shift_validity(params):
    """
    Check to see if the input time shifting params are valid
    Args:
        params: dictionary of parameters
    """
    for col in params["cols"]:
        if not params["cols"][col] % params["data_frame_frequency"] == 0:
            raise ValueError(
                f"{params['cols'][col]} is not a valid shift for your data frequency"
            )


def time_shift(params: dict, data: pd.DataFrame) -> pd.DataFrame:
    """
    Take input data and shift it back/forwards in time
    as dictated by the users parameters file.
    Args:
        params: dictionary of parameters
        data: input dataframe that will have key columns shifted
    Returns:
        time grid
    """
    timestamp_col = params.get("timestamp_col", "timestamp")

    # Check to ensure that all time shifts given are valid for the input dataframe
    _check_time_shift_validity(params)

    # set datetimeindex & enforce freq to n mins (resampling and offset ?)
    data = data.set_index(pd.to_datetime(data[timestamp_col]))
    data = data.asfreq("{}Min".format(params["data_frame_frequency"]))

    # Group columns by timeshifts to only shift once per timeshift.
    grouped = groupby(params["cols"].items(), key=lambda x: x[1])

    for _timeshift, column_tuples in grouped:
        cols = [x[0] for x in column_tuples]
        data[cols] = data[cols].shift(
            periods=int(_timeshift) // params["data_frame_frequency"]
        )

    data[timestamp_col] = data.index
    return data.reset_index(drop=True)
