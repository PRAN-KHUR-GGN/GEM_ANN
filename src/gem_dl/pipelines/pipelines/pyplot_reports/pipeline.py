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
EDA Pipeline
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_pdf, generate_notebook


def create_pipeline(**kwargs):
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
                    func=generate_pdf,
                    inputs=["params:pyplot_reports.pdf", *kwargs["plots"]],
                    outputs=None,
                    name=f"{name_prefix}generate_pdf",
                ),
                node(
                    func=generate_notebook,
                    inputs=["params:pyplot_reports.notebook", *kwargs["plots"]],
                    outputs=None,
                    name=f"{name_prefix}generate_notebook",
                ),
            ]
        )
    )
