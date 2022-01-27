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
    plot_correlation,
    plot_scatter,
    plot_density,
    plot_box,
    plot_bar,
    plot_actual_vs_residual,
    plot_actual_vs_predicted,
    plot_timeline,
    plot_timeline_column,
    plot_string,
    plot_pandas_df,
    plot_partial_dependency,
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
                    func=plot_correlation,
                    inputs={
                        "data": "correlation_data",
                        "params": "params:correlation_plot",
                    },
                    outputs={"correlation_plot": "correlation_plot"},
                    name=f"{name_prefix}correlation_plot",
                ),
                node(
                    func=plot_scatter,
                    inputs={"data": "scatter_data", "params": "params:scatter_plot"},
                    outputs={"scatter_plot": "scatter_plot"},
                    name=f"{name_prefix}scatter_plot",
                ),
                node(
                    func=plot_density,
                    inputs={"data": "density_data", "params": "params:density_plot"},
                    outputs={"density_plot": "density_plot"},
                    name=f"{name_prefix}density_plot",
                ),
                node(
                    func=plot_box,
                    inputs={"data": "box_data", "params": "params:box_plot"},
                    outputs={"box_plot": "box_plot"},
                    name=f"{name_prefix}box_plot",
                ),
                node(
                    func=plot_bar,
                    inputs={"params": "params:bar_plot", "data": "bar_data"},
                    outputs={"bar_plot": "bar_plot"},
                    name=f"{name_prefix}bar",
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
                    func=plot_actual_vs_residual,
                    inputs={
                        "params": "params:actual_vs_residual_plot",
                        "data": "actual_vs_residual_data",
                    },
                    outputs={"actual_vs_residual_plot": "actual_vs_residual_plot"},
                    name=f"{name_prefix}actual_vs_residual",
                ),
                node(
                    func=plot_timeline,
                    inputs={"params": "params:timeline_plot", "data": "timeline_data"},
                    outputs={"timeline_plot": "timeline_plot"},
                    name=f"{name_prefix}timeline",
                ),
                node(
                    func=plot_string,
                    inputs={
                        "params": "params:string_plot",
                        "string_data": "string_data",
                    },
                    outputs={"string_plot": "string_plot"},
                    name=f"{name_prefix}string_plot",
                ),
                node(
                    func=plot_pandas_df,
                    inputs={
                        "params": "params:pandas_df_plot",
                        "data": "pandas_df_data",
                    },
                    outputs={"pandas_df_plot": "pandas_df_plot"},
                    name=f"{name_prefix}pandas_df",
                ),
                node(
                    func=plot_partial_dependency,
                    inputs={
                        "params": "params:partial_dependency_plot",
                        "data": "partial_dependency_data",
                        "model": "partial_dependency_model",
                    },
                    outputs={"partial_dependency_plot": "partial_dependency_plot"},
                    name=f"{name_prefix}partial_dependency_plot",
                ),
                node(
                    func=plot_timeline_column,
                    inputs={
                        "params": "params:timeline_column",
                        "data": "timeline_column_data",
                    },
                    outputs={"timeline_column_plot": "timeline_column_plot"},
                    name=f"{name_prefix}timeline_column_plot",
                ),
            ]
        )
    )
