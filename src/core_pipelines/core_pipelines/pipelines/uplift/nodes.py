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
 Reporting of optimization results.
"""
import logging
from typing import Dict

import pandas as pd


logger = logging.getLogger(__name__)


def create_bulk_result_tables(
    params: Dict, recommendations: pd.DataFrame, opt_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Creates a bulk-optimization output table.

    Args:
        params: dictionary of parameters
        recommendations: output of bulk optimization
        opt_df: opt data
    Returns:
        Dataframes for states, controls, outcomes, constraints
    """

    extract_states = _extract_state(recommendations)
    extract_constraints_info = _extract_constraints_info(recommendations)
    extract_output = _extract_output(params, recommendations, opt_df)
    extract_controls = _extract_controls(recommendations)

    return {
        "states": extract_states,
        "controls": extract_controls,
        "outcomes": extract_output,
        "constraints": extract_constraints_info,
    }


def _extract_constraints_info(recommend_output: pd.DataFrame):
    """
    Extracts Penalty and Slack information from recommendation and returns a dataframe
    """
    if "penalties" not in recommend_output.columns:
        return pd.DataFrame()

    penalty_df = _convert_to_df("penalties", recommend_output)
    slack_df = _convert_to_df("slack", recommend_output)

    return penalty_df.join(slack_df, how="left")


def _extract_state(recommend_output: pd.DataFrame):
    """
    Extracts state information from recommendations and returns a dataframe
    """
    return _convert_to_df("state", recommend_output)


def _extract_output(params: Dict, recommend_output: pd.DataFrame, opt_df: pd.DataFrame):
    """
    Extracts outputs information from recommendations and returns a dataframe
    """
    target = params.get("target", "model_target")

    # append true opt set outcomes to output
    actual_df = opt_df[["run_id", target]].set_index("run_id")
    actual_df.columns = pd.MultiIndex.from_tuples([(target, "actual")])

    output_df = _convert_to_df("outputs", recommend_output).rename(
        columns={"outputs": target}
    )

    output_df = output_df.join(actual_df, how="left")

    # calculate deltas
    output_df[(target, "optimized_vs_predicted")] = (
        output_df[(target, "pred_optimized")] - output_df[(target, "pred_current")]
    )
    output_df[(target, "optimized_vs_predicted_pct")] = (
        output_df[(target, "optimized_vs_predicted")]
        / output_df[(target, "pred_current")]
        * 100
    )
    output_df[(target, "optimized_vs_actual")] = (
        output_df[(target, "pred_optimized")] - output_df[(target, "actual")]
    )

    output_df[(target, "optimized_vs_actual_pct")] = (
        output_df[(target, "optimized_vs_actual")] / output_df[(target, "actual")] * 100
    )

    return output_df


def _extract_controls(recommend_output: pd.DataFrame):
    """
    Extracts controls information from recommendations and returns a dataframe
    """
    return _convert_to_df("controls", recommend_output)


def _convert_to_df(col: str, recommend_output: pd.DataFrame):
    """
    Converts to a well formatted dataframe of the specified column
    """
    output, data_controls = _extract_df_with_multi_index(col, recommend_output)

    if col in ["state", "outputs"]:
        return output.join(data_controls, how="left")
    processed_df = _process_one_level_deep(col, data_controls)
    return output.join(processed_df, how="left")


def _process_one_level_deep(col: str, data_output: pd.DataFrame):
    """
    Extracts second level information from recommendations json and returns a dataframe
    """
    if data_output.empty:
        return data_output
    output_list = []
    for names in data_output.columns.levels[1]:
        # Replace missing values to a dict format (similar
        # to other cells in the dataframe)
        data_output_list = [
            value if pd.notnull(value) else {"current": None, "suggested": None}
            for value in data_output[col][names].tolist()
        ]
        sub_data = pd.DataFrame(data_output_list)
        sub_data.columns = pd.MultiIndex.from_product(
            [[col], [(names, col) for col in sub_data.columns]]
        )
        output_list.append(sub_data)
    processed_output = pd.concat(output_list, axis=1).set_index(data_output.index)

    return processed_output


def _extract_df_with_multi_index(col: str, recommend_output: pd.DataFrame):
    """
    Converts to a well formatted dataframe and multi-index the columns
    """
    data = recommend_output.copy()
    output = data[["run_id"]].set_index("run_id")
    output.columns = pd.MultiIndex.from_arrays([output.columns, output.columns])
    data_extract = pd.DataFrame(data[col].tolist())
    data_extract.columns = pd.MultiIndex.from_product([[col], data_extract.columns])
    data_extract.index = data["run_id"]

    return output, data_extract


def create_bulk_report(
    params: Dict,
    states: pd.DataFrame,
    controls: pd.DataFrame,
    outcomes: pd.DataFrame,
    constraints: pd.DataFrame,
    opt_df,
) -> pd.DataFrame:
    """
    Combines all four bulk optimization components into one single dataframe.

    Args:
        states: State variables dataframe from the bulk optimization output
        controls: Control variables dataframe from the bulk optimization output
        outcomes: Outcome variables dataframe from the bulk optimization output
        constraints: Constraints variables dataframe from the bulk
         optimization output

    Returns:
        Combined dataframe
    """
    return _merge_data(params, opt_df, constraints, controls, outcomes, states)


def _merge_data(
    params: Dict,
    opt_df: pd.DataFrame,
    constraints: pd.DataFrame,
    controls: pd.DataFrame,
    outcomes: pd.DataFrame,
    states: pd.DataFrame,
):
    """
    Choose relevant columns and merge all dataframes
    """
    # Renaming columns to be consistent across all files
    idx = pd.IndexSlice
    target_info = outcomes.rename(
        columns={
            "pred_current": "predicted_current",
            "pred_optimized": "suggested",
            "actual": "current",
        }
    ).loc[:, idx[:, ["current", "suggested", "predicted_current"]]]

    # Dropping "delta" from the table. Keeping only `current` and `suggested` values
    controls_info = controls.drop(
        [
            col_name
            for col_name in controls.columns.get_level_values(1)
            if "delta" in col_name
        ],
        axis=1,
        level=1,
    )

    # Getting timestamp for each run_id
    timestamp_info = opt_df[["run_id", "timestamp"]].set_index(["timestamp", "run_id"])

    # Create one dataframe
    joined_df = (
        states.join(controls_info, how="left")
        .join(constraints, how="left")
        .join(target_info, how="left")
    )
    re_index_data = _reindex_data(joined_df, params)
    re_index_data_with_timestamp = re_index_data.join(
        timestamp_info, how="left"
    ).reorder_levels(["run_id", "timestamp", "type", "variable"])
    re_index_data_with_timestamp = re_index_data_with_timestamp[
        ["current", "suggested", "predicted_current"]
    ]
    return re_index_data_with_timestamp


def _reindex_data(joined_df: pd.DataFrame, params: Dict):
    """
    Convert to a three level multi-index columns and re-format indices
    """
    data_indexed = joined_df.copy()
    new_index = []
    target = params.get("target", "model_target")
    for main in data_indexed.columns.get_level_values(0).unique():
        for name in data_indexed[main].columns.get_level_values(0):
            if main == target:
                new_index.append(("target", main, name))
            elif main in ["controls", "penalties", "slack"]:
                new_index.append((main, *name))
            else:
                new_index.append((main, name, "current"))
    data_indexed.columns = pd.MultiIndex.from_tuples(
        new_index, names=("type", "variable", "measure")
    )
    data_reformat = data_indexed.stack([0, 1])
    return data_reformat
