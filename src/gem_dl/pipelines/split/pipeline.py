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
Data Splitting Pipeline
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data


def create_pipeline():
    """"
    Using this pipeline expects that the users' parameters includes a block
    with the following example structure ::

        split:
            common: &split_common
                type: frac
                train_split_fract: 0.9 # only when `type: frac`
                datetime_val: !!timestamp '2001-04-26 03:59:59' # only if `type: date`
            in_out:
                <<: *split_common
                datetime_col: "status_time"
            mill:
                <<: *split_common
                datetime_col: "time_col"


    """

    return pipeline(
        pipe=Pipeline(
            [
                node(
                    func=split_data,
                    inputs=dict(params="params:split", data="data"),
                    outputs=dict(train="train_data", test="test_data"),
                    name="split_data",
                )
            ]
        )
    )
