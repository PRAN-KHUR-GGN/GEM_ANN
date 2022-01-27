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
Plotting Pipeline
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    plot_scatter,
    plot_box,
    plot_actual_vs_predicted,
    plot_timeline,
    plot_single_feature_overview,
)


def create_pipeline(**kwargs):
    """
    Create a pipeline of all plotting nodes

    Returns:
        Pipeline
    """

    name_prefix = f"{kwargs['name_prefix']}." if "name_prefix" in kwargs else ""

    return pipeline(
        pipe=Pipeline(
            nodes=[
                node(
                    func=plot_scatter,
                    inputs={"data": "scatter_data", "params": "params:scatter_plot"},
                    outputs={"scatter_plot": "scatter_plot"},
                    name=f"{name_prefix}scatter_plot",
                ),
                node(
                    func=plot_box,
                    inputs={"data": "box_data", "params": "params:box_plot"},
                    outputs={"box_plot": "box_plot"},
                    name=f"{name_prefix}box_plot",
                ),
                node(
                    func=plot_actual_vs_predicted,
                    inputs={
                        "params": "params:actual_vs_predicted_plot",
                        "data": "actual_vs_predicted_data",
                    },
                    outputs={"actual_vs_predicted_plot": "actual_vs_predicted_plot"},
                    name=f"{name_prefix}actual_vs_predicted",
                ),
                node(
                    func=plot_timeline,
                    inputs={"params": "params:timeline_plot", "data": "timeline_data"},
                    outputs={"timeline_plot": "timeline_plot"},
                    name=f"{name_prefix}timeline",
                ),
                node(
                    func=plot_single_feature_overview,
                    inputs={"params": "params:feature_overview", "data": "data"},
                    outputs={"feature_overview": "feature_overview"},
                    name=f"{name_prefix}overview",
                ),
            ]
        )
    )
