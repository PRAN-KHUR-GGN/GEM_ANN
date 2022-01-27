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
Nodes of the data stitching pipeline.
"""
import logging
from functools import reduce
import pandas as pd

logger = logging.getLogger(__name__)


def stitch(params: dict, *dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Stitches multiple dataframes together from a list
    Args:
        params: Parameters dict
        *dfs: dfs to be merged
    Returns:
        df_merged: Merged data
    """

    # Iterate through each dataframe in input df list.
    # For each dataframe, perform an outer join on the designated key.
    # Repeat until we have iterated through all dataframes within the list.

    df_merged = reduce(
        lambda left, right: pd.merge(
            left, right, on=params["join_cols"], how=params["join_type"]
        ),
        dfs,
    )
    return df_merged
