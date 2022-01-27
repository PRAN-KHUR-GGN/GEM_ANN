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


from typing import Dict, List
from kedro.pipeline import Pipeline, pipeline


def namespace_pipelines(
    pipelines: Dict[str, Pipeline], namespace: str, ignore: List[str] = None
):

    if not ignore:
        ignore = []

    return {
        name
        if name in ignore
        else f"{namespace}.{name}": pipeline(
            pipe=p,
            parameters={
                param: param.replace("params:", f"params:{namespace}.")
                for param in [
                    i
                    for node in p.nodes
                    for i in node.inputs
                    if i.startswith("params:") and i != "params:KEDRO_ENV"
                ]
            },
            namespace=namespace,
        )
        for name, p in pipelines.items()
    }


def merge_pipelines(pipelines: Dict[str, Pipeline]) -> Pipeline:

    merged_pipeline = Pipeline([pl for _, pl in pipelines.items()])
    return merged_pipeline
