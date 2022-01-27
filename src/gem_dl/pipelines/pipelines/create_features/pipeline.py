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
Feature Creation Training Pipeline
"""
from typing import List, Callable
from kedro.pipeline import Pipeline, node, pipeline

from optimus_core.core.utils import generate_run_id

from .grid import create_time_grid
from .static import create_train_features, create_opt_features, merge_to_grid


def create_pipeline(input_data: List[str], method: Callable):
    """
    In addition to the below arguments, this pipeline expects the user has
    parameters with structure like this ::

        create_features:
            pipeline_timezone: "${pipeline_timezone}" # see globals
            n_jobs: 6
            grid:
                frequency: "1H"
                offset_start: "2H"
                offset_end: "2H"

    Args:
        input_data:
        method:

    Returns:

    """

    if method not in {create_train_features, create_opt_features}:
        raise ValueError("unknown method")

    return pipeline(
        pipe=Pipeline(
            [
                node(
                    create_time_grid,
                    ["params:create_features", *input_data],
                    "time_grid",
                    name="create_time_grid",
                ),
                *[
                    node(
                        method,
                        ["params:create_features", "td", data, "time_grid"],
                        f"{data}.static_features",
                        name=f"create_features_{data}",
                    )
                    for data in input_data
                ],
                node(
                    merge_to_grid,
                    ["time_grid", *[f"{data}.static_features" for data in input_data]],
                    "static_features",
                    name="merge_to_grid",
                ),
            ]
        )
    )


def create_run_id_pipeline():
    return pipeline(
        pipe=Pipeline(
            [node(generate_run_id, "input", "output", name="generate_run_id")]
        )
    )
