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
Data Export Pipeline
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    export_runs,
    export_tags,
    export_states,
    export_recommendations,
    export_predictions,
    export_control_sensitivities,
)


def create_pipeline():

    return pipeline(
        Pipeline(
            [
                node(
                    export_runs,
                    dict(
                        iso_format="params:export.iso_format",
                        connection_endpoint="params:export.connection_endpoint",
                        data="input_features",
                    ),
                    None,
                    name="export_runs",
                ),
                node(
                    export_tags,
                    dict(
                        connection_endpoint="params:export.connection_endpoint", td="td"
                    ),
                    None,
                    name="export_tags",
                ),
                node(
                    export_states,
                    dict(
                        iso_format="params:export.iso_format",
                        connection_endpoint="params:export.connection_endpoint",
                        td="td",
                        data="input_features",
                    ),
                    None,
                    name="export_states",
                ),
                node(
                    export_recommendations,
                    dict(
                        connection_endpoint="params:export.connection_endpoint",
                        recommendation_list="input_recommendations",
                    ),
                    None,
                    name="export_recommendation_to_optimus_API",
                ),
                node(
                    export_predictions,
                    dict(
                        connection_endpoint="params:export.connection_endpoint",
                        prediction_list="input_predictions",
                    ),
                    None,
                    name="export_predictions_to_optimus_API",
                ),
                node(
                    export_control_sensitivities,
                    dict(
                        connection_endpoint="params:export.connection_endpoint",
                        control_sensitivity_list="input_cs",
                    ),
                    None,
                    name="export_control_sensitivities_to_optimus_API",
                ),
            ]
        )
    )
