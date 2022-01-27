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

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..pyplot_charts.nodes import plot_string, plot_pandas_df, plot_timeline, plot_box
from ..pyplot_reports.nodes import generate_pdf


def generate_uplift_report_figures(
    params: Dict,
    output_data: pd.DataFrame,
    controls_data: pd.DataFrame,
    combined_bulk_data: pd.DataFrame,
):
    """
    Generate a PDF Uplift Report
    Args:
        params: params for pyplot_charts.generate_pdf
        output_data: the output data to report on
        controls_data: the controls data to report on
        combined_bulk_data: The combined recommendation dataframe

    Returns:
        Dictionary of plots found in uplift report pdf
    """

    controls_nested = controls_data.copy(deep=True)
    controls_nested.columns = pd.MultiIndex.from_tuples(
        controls_nested.columns.droplevel()
    )
    controls = sorted(list(set(controls_nested.columns.get_level_values(0))))
    flattened_recs = _normalized_rec_deltas(combined_bulk_data)
    model_target = params.get("target", "model_target")
    pdf_pages = {
        "Results.png": _plot_results_text(output_data, model_target),
        "Details.png": _plot_df(output_data.filter(like="_vs_").describe(), "Details"),
        "Timeline.png": _plot_details_timeline(output_data, model_target),
        "Controls.png": _plot_df(controls_nested.describe(), "Controls"),
        "Controls Distribution.png": _plot_rec_distribution(
            flattened_recs, "normalized_delta"
        ),
        **{
            f"Control {control}.png": _plot_current_and_suggested(
                controls_nested, control
            )
            for control in controls
        },
        **{
            f"Raw Control {control} Deltas.png": _plot_rec_distribution(
                flattened_recs[flattened_recs.controllable_variable == control]
            )
            for control in controls
        },
    }

    return pdf_pages


def generate_uplift_report(params: Dict, uplift_report_figures: Dict[str, plt.Figure]):
    generate_pdf(params, *uplift_report_figures.values())


def _plot_results_text(data: pd.DataFrame, target: str):

    output = (
        "Average uplift against predicted was {:.3f}, "
        "or {:.2f}% (median).\n\n".format(
            data[(target, "optimized_vs_predicted")].mean(),
            data[(target, "optimized_vs_predicted_pct")].median(),
        )
    )

    output += "Average uplift against actual was {:.3f}, or {:.2f}% (median).\n".format(
        data[(target, "optimized_vs_actual")].mean(),
        data[(target, "optimized_vs_actual_pct")].median(),
    )

    return plot_string(
        {"title": "The goal was to maximize {}.".format(target)}, output
    )["string_plot"]


def _plot_df(data: pd.DataFrame, title: str):

    return plot_pandas_df({"title": title}, data)["pandas_df_plot"]


def _plot_details_timeline(data: pd.DataFrame, target: str):

    data.columns = data.columns.droplevel()

    return plot_timeline(
        {
            "title": "Uplift Simulation Timeline",
            "timestamp_col": "run_id",
            "columns": ["actual", "pred_current", "pred_optimized"],
            "ylabel": target,
        },
        data.reset_index(),
    )["timeline_plot"]


def _plot_current_and_suggested(ctrl_df, ctrl):
    """ Plot current and suggested control values """

    plt.close("all")

    sub_df = ctrl_df[ctrl]

    x = np.zeros((len(ctrl_df), 2))
    x[:, 1] = 1
    y = sub_df[["current", "suggested"]].values

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(x.T, y.T)
    plt.xticks(ticks=[0, 1], labels=["current", "suggested"])
    plt.ylabel(ctrl)
    plt.title(f"{ctrl} suggestions")

    return fig


def _normalized_rec_deltas(combined_bulk_data):
    """Subset bulk report to control recs.
    Calculate raw and studentized deltas in recs
    """
    # combined_bulk_data has a multi-index
    data = combined_bulk_data.loc[
        (slice(None), slice(None), ["controls"], slice(None)), :
    ]
    data["delta"] = data["suggested"] - data["current"]

    data.reset_index(inplace=True)
    data.drop(
        columns=["run_id", "timestamp", "type", "predicted_current"], inplace=True
    )

    agg_delta = (
        data.groupby("variable")["delta"]
        .agg(np.std)
        .reset_index()
        .rename(columns={"delta": "delta_std"})
    )
    merged_data = data.merge(agg_delta, how="left", on="variable")
    merged_data.rename(columns={"variable": "controllable_variable"}, inplace=True)
    merged_data["normalized_delta"] = merged_data["delta"] / merged_data["delta_std"]
    return merged_data


def _plot_rec_distribution(recs_df, delta_col="delta"):
    """Plot distribution (box-plot) of recommended changes for
    a given control"""

    plt.close("all")
    return plot_box(
        recs_df,
        {
            "title": "Dist. of Recommended Control Changes",
            "by": {"column": "controllable_variable"},
            "column": delta_col,
        },
    )["box_plot"]
