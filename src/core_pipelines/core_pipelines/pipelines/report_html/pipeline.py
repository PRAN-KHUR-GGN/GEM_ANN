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
Pipeline to create an html report
"""

from kedro.pipeline import Pipeline, node
from .nodes import generate_report


def create_pipeline():
    return Pipeline(
        [
            node(
                generate_report,
                inputs={"figures": "figures", "params": "params"},
                outputs=None,
                name="generate_report",
            )
        ]
    )
