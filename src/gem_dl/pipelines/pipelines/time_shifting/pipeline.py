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
Data Source Time Shift Pipeline
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import time_shift


def create_pipeline():
    return pipeline(
        pipe=Pipeline(
            [
                node(
                    time_shift,
                    dict(params="params:time_shifting", data="input_data"),
                    "time_shifted_data",
                    name="time_shift",
                )
            ]
        )
    )
