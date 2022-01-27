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

from kedro.pipeline import Pipeline, pipeline, node
from ..pyplot_reports.nodes import generate_pdf_batch_analytics
from . import profile, sensor_trend, feature_overview


def create_pipeline(**kwargs):  # pylint:disable=unused-argument
    """
    Using this pipeline expects users to have a parameters file that includes blocks
    like::

        feature_overview_report:
            target_column: filter_trough_lvl_increase_filter
            hue_column: reactor
            start_column: reactor_start_time
            feature_columns: []
            pdf:
                output_path: data/results/feature_overview
                report_name: feature_overview
                timestamp: True
                title: "Feature overview"
                author: "My Name / CST Name"
                subject: "Latest outputs"
                add_bookmarks: True
                # list of features with bookmarks; if empty add bookmarks to all
                features_to_bookmark: []

            profile_report:
            sensors_to_plot:
                - reactor_temp
                - reactor_agitator_speed
                - reactor_P
                - filter_trough_lvl
                - filter_infeed
            times_to_mark:
                - reactor_end_timestep
                - filter_start_timestep
            time_unit: time_step
            batch_id_col: batch_id
            start_time_scope: null
            end_time_scope: null
            q: 0.05
            nb_line_per_plot: 3
            datetime_name: datetime
            modular_title: null
            figsize: [16, 9]
            pdf:
                output_path: data/results/batch_profiles
                report_name: batch_profiles
                timestamp: True
                title: "Batch profiles"
                author: "My Name / CST Name"
                subject: "Latest outputs"

            sensor_range_reactor_report:
            sensors_to_plot:
                - reactor_temp
                - reactor_agitator_speed
                - reactor_P
            time_unit: time_step
            batch_id_col: batch_id
            trend_type: range
            hue: reactor
            alpha: null
            linewidth: null
            figsize: [16, 9]
            x_lim_quantile: 0.9
            pdf:
                output_path: data/results/sensor_trend
                report_name: sensor_range_reactor
                timestamp: True
                title: "Sensor range reactor"
                author: "My Name / CST Name"
                subject: "Latest outputs"

            sensor_overlay_filter_report:
            target: 'filter_trough_lvl_increase_filter'
            sensors_to_plot:
                - filter_trough_lvl
                - filter_infeed
            time_unit: null
            batch_id_col: batch_id
            trend_type: overlay
            hue: batch_type
            alpha: 0.1
            linewidth: null
            figsize: [16, 9]
            x_lim_quantile: 0.9
            pdf:
                output_path: data/results/sensor_trend
                report_name: sensor_overlay_filter
                timestamp: True
                title: "Sensor overlay filter"
                author: "My Name / CST Name"
                subject: "Latest outputs"

    Returns:
        Pipeline
    """

    return pipeline(
        pipe=Pipeline(
            nodes=[
                node(
                    func=profile.plot_all_batches_profiles,
                    inputs={
                        "batches_header": "batches_header",
                        "batches_time_series": "batches_time_series",
                        "sensors_to_plot": "params:pyplot_batch.sensors_to_plot",
                        "times_to_mark": "params:pyplot_batch.times_to_mark",
                        "start_time_scope": "params:pyplot_batch.start_time_scope",
                        "end_time_scope": "params:pyplot_batch.end_time_scope",
                        "time_unit": "params:pyplot_batch.time_unit",
                        "batch_id_col": "params:pyplot_batch.batch_id_col",
                        "q": "params:pyplot_batch.q",
                        "nb_line_per_plot": "params:pyplot_batch.nb_line_per_plot",
                        "datetime_col": "params:pyplot_batch.datetime_col",
                        "modular_title": "params:pyplot_batch.modular_title",
                        "figsize": "params:pyplot_batch.figsize",
                    },
                    outputs={"all_batch_profiles": "all_batch_profiles"},
                    name="plot_all_batches_profiles",
                ),
                node(
                    func=sensor_trend.plot_all_sensor_trends,
                    inputs={
                        "batches_header": "batches_header",
                        "batches_time_series": "batches_time_series",
                        "sensors_to_plot": "params:pyplot_batch.sensors_to_plot",
                        "time_unit": "params:pyplot_batch.time_unit",
                        "batch_id_col": "params:pyplot_batch.batch_id_col",
                        "trend_type": "params:pyplot_batch.trend_type",
                        "hue": "params:pyplot_batch.hue",
                        "alpha": "params:pyplot_batch.alpha",
                        "linewidth": "params:pyplot_batch.linewidth",
                        "figsize": "params:pyplot_batch.figsize",
                        "x_lim_quantile": "params:pyplot_batch.x_lim_quantile",
                    },
                    outputs={"all_sensor_trends": "all_sensor_trends"},
                    name="plot_all_sensor_trends",
                ),
                node(
                    func=feature_overview.plot_all_features_overview,
                    inputs={
                        "feature_dataframe": "feature_dataframe",
                        "target_column": "params:pyplot_batch.target_column",
                        "hue_column": "params:pyplot_batch.hue_column",
                        "start_column": "params:pyplot_batch.start_column",
                        "feature_columns": "params:pyplot_batch.feature_columns",
                    },
                    outputs={"all_features_overview": "all_features_overview"},
                    name="plot_all_features_overview",
                ),
            ]
        )
    )


def create_pipeline_pdf_report(**kwargs):
    """
    Create a pipeline of reporting nodes
    Args:
        **kwargs:

    Returns:
        Pipeline
    """

    name_prefix = f"{kwargs['name_prefix']}." if "name_prefix" in kwargs else ""

    return pipeline(
        pipe=Pipeline(
            nodes=[
                node(
                    func=generate_pdf_batch_analytics,
                    inputs=[
                        "params:pyplot_batch.pyplot_reports.pdf",
                        "partitioned_dataset",
                    ],
                    outputs=None,
                    name=f"{name_prefix}generate_pdf",
                )
            ]
        )
    )
