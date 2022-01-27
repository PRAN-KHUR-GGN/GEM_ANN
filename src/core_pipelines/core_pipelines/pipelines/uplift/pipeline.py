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

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_bulk_result_tables, create_bulk_report
from .reports import generate_uplift_report, generate_uplift_report_figures


def create_pipeline():  # pylint:disable=unused-argument
    return pipeline(
        pipe=Pipeline(
            [
                node(
                    create_bulk_result_tables,
                    dict(
                        params="params:uplift",
                        recommendations="recommendations",
                        opt_df="input_data",
                    ),
                    dict(
                        states="bulk_state",
                        controls="bulk_ctrl",
                        outcomes="bulk_output",
                        constraints="bulk_constraints",
                    ),
                    name="create_bulk_result_tables",
                ),
                node(
                    create_bulk_report,
                    dict(
                        params="params:uplift",
                        states="bulk_state",
                        controls="bulk_ctrl",
                        outcomes="bulk_output",
                        constraints="bulk_constraints",
                        opt_df="input_data",
                    ),
                    "bulk_report",
                    name="create_bulk_report",
                ),
                node(
                    generate_uplift_report_figures,
                    dict(
                        params="params:uplift",
                        output_data="bulk_output",
                        controls_data="bulk_ctrl",
                        combined_bulk_data="bulk_report",
                    ),
                    "uplift_report_figures",
                    name="generate_uplift_report_figures",
                ),
                node(
                    generate_uplift_report,
                    dict(
                        params="params:uplift",
                        uplift_report_figures="uplift_report_figures",
                    ),
                    None,
                    name="generate_uplift_report",
                ),
            ]
        )
    )
