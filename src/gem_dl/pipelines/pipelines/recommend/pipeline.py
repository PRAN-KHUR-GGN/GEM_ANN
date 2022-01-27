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
Optimization Pipeline
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    bulk_optimize,
    get_feature_list,
    get_penalties,
    get_repairs,
    get_solver_bounds,
)


def create_pipeline():  # pylint:disable=unused-argument
    """
    This pipeline's main node is the last one, `bulk_optimize_*`, which parallelizes
    the optimization of a given shift, to yield counterfactual recommendations.

    The preceeding 4 nodes are meant to be over-ridden, and exist as convenience
    utilities. They each create a single artifact/input needed by the bulk_optimize
    nodes. Each of those nodes can be replaced, provided they produce artifacts with
    the schema described below:

    * features_list (get_feature_list): This is a list of raw input tag names
      that the optimization must be run over.
    * solver_bounds (get_solver_bounds): This is a dictionary of bounds objects,
      keyed by the index of `input_data`.
    * repairs_dict (get_repairs): This is a dictionary of lists of repairs, keyed
      by the index of `input_data`.
    * penalties_dict (get_penalties): This is a dictionary of lists of penalties,
      keyed by the index of `input_data`.

    """
    return pipeline(
        pipe=Pipeline(
            [
                node(
                    get_feature_list,
                    inputs=dict(model="input_model", data="input_data"),
                    outputs="features_list",
                    name="initial_feature_list",
                ),
                node(
                    get_solver_bounds,
                    inputs=dict(
                        input_data="input_data", td="td", features="features_list"
                    ),
                    outputs="bounds_dict",
                    name="create_bounds_dictionary",
                ),
                node(
                    get_repairs,
                    inputs=dict(
                        input_data="input_data", td="td", features="features_list"
                    ),
                    outputs="repairs_dict",
                    name="create_repairs_dictionary",
                ),
                node(
                    get_penalties,
                    inputs=dict(input_data="input_data"),
                    outputs="penalties_dict",
                    name="create_penalties_dictionary",
                ),
                node(
                    func=bulk_optimize,
                    inputs=dict(
                        params="params:recommend",
                        td="td",
                        data="input_data",
                        model="input_model",
                        bounds_dict="bounds_dict",
                        repairs_dict="repairs_dict",
                        penalty_dict="penalties_dict",
                        features="features_list",
                    ),
                    outputs=dict(
                        recommendations="recommendations",
                        recommended_controls="recommended_controls",
                        projected_optimization="projected_optimization",
                        latest_problem="latest_problem",
                    ),
                    name="bulk_optimize",
                ),
            ]
        )
    )
