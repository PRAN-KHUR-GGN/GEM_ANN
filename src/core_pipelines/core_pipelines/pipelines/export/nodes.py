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
Nodes of the data export pipeline.
"""
from typing import List
import itertools
import uuid

import pandas as pd

from optimus_core.core.tag_management import TagDict
from optimus_core.core.utils import generate_run_id
from . import optimus_api


def _get_states(td: TagDict) -> List[str]:
    """ return current state feature columns """
    return td.select("model_feature") + td.select("model_target")


def export_runs(iso_format: str, connection_endpoint: str, data: pd.DataFrame) -> None:
    """
    Exports each timestamp entry along with its generated run_id to the runs
    table in the Optimus API
    Args:
        connection_endpoint: Connection string for REST endpoint
        data: DataFrame containing Timestamp and Run ID

    Returns:
        None
    """

    # cast all timestamps from Timestamp objects to strings
    data["timestamp"] = data["timestamp"].dt.strftime(iso_format)

    runs_df: pd.DataFrame = data[["run_id", "timestamp"]].copy()
    runs_df.rename(columns={"run_id": "id"}, inplace=True)
    runs = runs_df.to_dict(orient="records")
    client = optimus_api.Client(connection_endpoint)
    client.create_runs(runs)


def export_tags(connection_endpoint: str, td: TagDict) -> None:
    """
    Exports the tag dictionary to the Optimus API
    Args:
        connection_endpoint: Connection string for REST endpoint
        td: TagDict

    Returns:
        None
    """
    tags_df: pd.DataFrame = td.to_frame()
    if "area" not in tags_df.columns:
        tags_df["area"] = None
    tags_df = tags_df[["tag", "name", "area", "unit"]]
    tags_df.rename(columns={"tag": "id", "name": "clear_name"}, inplace=True)
    tags = tags_df.to_dict(orient="records")
    client = optimus_api.Client(connection_endpoint)
    client.create_tags(tags)


def export_states(
    iso_format: str, connection_endpoint: str, td: TagDict, data: pd.DataFrame
) -> None:
    """
    Exports current value of all features into states table of Optimus API
    Args:
        connection_endpoint: Connection string for REST endpoint
        td: TagDict
        data: Input Dataset on which Optimization was run

    Returns:

    """
    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.strftime(iso_format)
    state_cols = _get_states(td)
    states_df = data[["run_id", *state_cols]].copy()
    states_df = pd.melt(states_df, id_vars=["run_id"], var_name="tag_id")
    states_df = generate_run_id(states_df, "id")
    states = states_df.to_dict(orient="records")
    client = optimus_api.Client(connection_endpoint)
    client.create_states(states)


def export_recommendations(
    connection_endpoint: str, recommendation_list: pd.DataFrame
) -> None:
    """
    Exports newly suggested value for each run per optimizable feature to the
     Optimus API

    Args:
        connection_endpoint: Connection string for REST endpoint
        recommendation_list: List (for each run) of List (for each feature)
         of Dictionaries

    Returns:
        None
    """
    recommendation_list = recommendation_list.to_dict(orient="list").values()
    merged_recommendations = [
        recommendation
        for recommendation in list(itertools.chain.from_iterable(recommendation_list))
        if recommendation
    ]
    for recommendation in merged_recommendations:
        recommendation["id"] = str(uuid.uuid4())
        recommendation["status"] = "Pending"
        recommendation["comment"] = "none"

    client = optimus_api.Client(connection_endpoint)
    client.create_recommendations(merged_recommendations)


def export_predictions(
    connection_endpoint: str, prediction_list: pd.DataFrame
) -> None:  # List[Dict]) -> None:
    """
    Exports the Projected Uplift (with optimization) as well as the
    current output (without optimization) to the Optimus API
    Args:
        connection_endpoint: Connection string for REST endpoint
        prediction_list: List of Predicted Results from Optimization Pipeline

    Returns:
        None
    """

    prediction_list = prediction_list.to_dict(orient="records")

    for prediction in prediction_list:
        prediction["id"] = str(uuid.uuid4())
    client = optimus_api.Client(connection_endpoint)
    client.create_predictions(prediction_list)


def export_control_sensitivities(
    connection_endpoint: str, control_sensitivity_list: pd.DataFrame
) -> None:  # List[Dict]) -> None:
    """
    Exports the Control Sensitivities
    Args:
       connection_endpoint: Connection string for REST endpoint
       control_sensitivity_list: List of control sensitivities
    Returns:
        None
    """

    control_sensitivities = control_sensitivity_list.to_dict(orient="records")

    client = optimus_api.Client(connection_endpoint)
    client.create_control_sensitivities(control_sensitivities)
