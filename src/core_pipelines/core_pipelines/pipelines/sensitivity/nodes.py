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
import numpy as np

from optimus_core.core.tag_management import TagDict
from optimizer.problem import StatefulContextualOptimizationProblem

logger = logging.getLogger(__name__)


def create_sensitivity_plot_data(
    params: Dict,
    td: TagDict,
    model,
    opt_df: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Generates the data to display sensitivity charts within the UI
    Args:
        params: Dict w/2 keys, a list of the unique ids defining each run (e.g.
        run_id and timestamp), and n_points, how many control values to
        plot objective values at.
        td: A tagdict object.
        model: A predictive model defining the objective.
        opt_df: Dataframe holding our optimization set.
        recommendations: List of dictionaries holding recommendations.

    Returns: Dictionary keyed by `sensitivity_plot_df`,
    holding the long-form sensitivity plot data.

    """

    recommendations = recommendations.to_dict(orient="records")

    return_list = [pd.DataFrame()] * len(recommendations)
    # Copy to avoid modifying original reference
    opt_df = opt_df.copy().set_index(params["unique_ids"])
    for rec_idx, rec in enumerate(recommendations):
        control_range_dict = {
            control: determine_control_range(td, control, params["n_points"])
            for control in rec["controls"].keys()
        }
        rec_return_list = []
        # .loc returns a Series, .to_frame().T converts to 1-row DF.
        opt_data_row = (
            opt_df.loc[tuple([rec[uid] for uid in params["unique_ids"]])].to_frame().T
        )

        # Set up the objective (with context) for evaluation.
        # Don't worry about repairs/penalties
        problem = _create_opt_problem(opt_df, rec, model, params)
        for control in rec["controls"]:
            # If we're looking at sensitivities to local percent changes,
            # update the control ranges
            if params.get("percentile_range"):
                control_range_dict[control] = _create_percentile_ranges(
                    params, opt_data_row[control].values, control_range_dict[control]
                )

            return_df = pd.DataFrame(
                np.repeat(
                    opt_data_row.values, control_range_dict[control].shape[0], axis=0
                ),
                columns=opt_data_row.columns,
            )
            return_df[control] = control_range_dict[control]
            return_df["target_value"] = problem.objective(return_df)

            return_df = (
                return_df[["target_value", control]]
                .rename(columns={control: "control_value"})
                .copy()
            )
            return_df["control_tag_id"] = control
            return_df["target_tag_id"] = td.select("model_target")[0]
            for uid in params["unique_ids"]:
                return_df[uid] = rec[uid]
            rec_return_list.append(return_df)
            # Store the intermediate result
        return_list[rec_idx] = pd.concat(rec_return_list, axis=0)

    return {"sensitivity_plot_df": pd.concat(return_list, axis=0, ignore_index=True)}


def _create_percentile_ranges(params, curr_control_value, control_sensitivity_array):
    raw_percentages = (
        np.linspace(*params["percentile_range"], num=params["n_points"]) / 100.0
    )
    relative_changes = curr_control_value * (1 + raw_percentages)

    relative_changes = np.minimum(np.max(control_sensitivity_array), relative_changes)
    return np.maximum(np.min(control_sensitivity_array), relative_changes)


def _create_opt_problem(opt_df, rec, model, params):
    try:
        window_size = (
            model.steps[-1][-1]
            .params.get("data_reformatter", {})
            .get("kwargs", {})
            .get("length", 1)
        )
    except AttributeError:
        window_size = 1

    idx = int(
        np.argwhere(opt_df.reset_index().run_id.values == rec["run_id"]).flatten()
    )

    problem = StatefulContextualOptimizationProblem(
        model,
        state=opt_df.iloc[[idx], :],
        context_data=opt_df.iloc[max(idx - window_size + 1, 0) : idx, :],
        optimizable_columns=rec["controls"].keys(),
        sense="maximize",
        objective_kwargs=params["objective_kwargs"],
    )
    return problem


def determine_control_range(
    td: TagDict, control: str, n_points: int = 50
) -> np.ndarray:
    """
    Determine the range over which we should generate the
    sensitivity plot for a given recommendation.
    Heavily depends upon the ranges and constraint
    sets present in the TagDict.

    Args:
        td: TagDictionary dataframe
        control: String identifying a tag.
        n_points: Number of points to plot

    Returns: 1-dimensional np.ndarray.

    """
    tag_list = td.select("tag", lambda x: True)
    # load params from config
    if control not in tag_list:
        raise ValueError(f"{control} is not in the tag_dict")
    td_row = td.to_frame()[td.to_frame().tag == control].copy()
    # If constraint_range is specified, we should use these
    if any(td_row["constraint_set"].notnull()):
        control_range = pd.eval(td_row["constraint_set"])[0].astype("int")
    # else look at op_min
    elif all(td_row["op_min"].notnull()) and all(td_row["op_max"].notnull()):
        control_range = np.arange(
            td_row["op_min"].values,
            td_row["op_max"].values,
            (td_row["op_max"] - td_row["op_min"]).values / n_points,
        )
    # Otherwise look at range_min/range_max
    else:
        control_range = np.arange(
            td_row["range_min"].values,
            td_row["range_max"].values,
            (td_row["range_max"] - td_row["range_min"]).values / n_points,
        )
    return control_range
