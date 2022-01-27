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
Core nodes performing the optimization.
"""
import logging
from ast import literal_eval
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd
from .utils import (
    ContextualOptimizationProblemFactory,
    StatefulOptimizationProblemFactory,
    get_controls,
    get_on_features,
    get_reformatter_property,
)
from optimizer.constraint import Repair, penalty, repair
from optimizer.problem import OptimizationProblem, StatefulContextualOptimizationProblem
from optimizer.solvers import Solver
from optimizer.stoppers.base import BaseStopper
from optimizer.utils.diagnostics import get_penalties_table, get_slack_table
from optimus_core.core.tag_management import TagDict
from optimus_core.core.utils import load_obj
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_penalty_slack(solutions: pd.DataFrame, penalties: List) -> pd.DataFrame:
    """
    Get the penalties and slacks for all constraints. Slack is calculated
    only for inequality constraints. Returns a dataframe.
    """
    if not penalties:
        return pd.DataFrame(index=solutions.index)
    penalty_table = get_penalties_table(solutions, penalties)
    slack_table = get_slack_table(solutions, penalties)
    return penalty_table.join(slack_table, how="left")


def _safe_run(func):
    """
    Decorator for handling keyboard interupt in optimize function
    Args:
        func: function to decorate

    Returns:
        Empty tuple or results from func
    """

    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.info(
                msg=f"Execution halted by user, raising KeyboardInterrupt."
                f"Was executing {args} and {kwargs}"
            )
            return ({}, {}, {})

    return func_wrapper


@_safe_run
def optimize(  # pylint:disable=too-many-locals
    timestamp: datetime,
    row: pd.DataFrame,
    ui_states: List[str],
    controls: List[str],
    on_controls: List[str],
    target: str,
    problem: OptimizationProblem,
    solver: Solver,
    stopper: BaseStopper,
) -> Tuple:
    """
    Optimizes a single row and returns the expected json object.
    Args:
        timestamp: timestamp key
        row: row of data to score
        ui_states: list of state columns to include in the results
        controls: list of controllable features
        on_controls: list of controllable features that are on
        target: target to optimize objective function
        problem: optimization problem
        solver: solver
        stopper: stopper
    Returns:
        Tuple of JSON results
    """
    # score with current and with optimal controls
    scores = pd.concat([row] * 2, ignore_index=True)
    scores.index = ["curr", "opt"]

    if solver:
        stop_condition = False
        while not stop_condition:
            parameters = solver.ask()
            obj_vals, parameters = problem(parameters)
            solver.tell(parameters, obj_vals)
            stop_condition = solver.stop()
            if stopper:
                stopper.update(solver)
                stop_condition |= stopper.stop()

        best_controls, _ = solver.best()
        scores.loc["opt", on_controls] = best_controls

    # Here we only evaluate the objective. This means penalties will not be applied.
    scores["_pred"] = problem.objective(scores)
    penalty_slack_table = _get_penalty_slack(scores, problem.penalties)
    states_ = {state: float(row[state].values[0]) for state in ui_states}
    controls_ = {
        ctrl: {
            "current": float(scores.loc["curr", ctrl]),
            "suggested": float(scores.loc["opt", ctrl]),
            "delta": float(scores.loc["opt", ctrl] - scores.loc["curr", ctrl]),
        }
        for ctrl in controls
    }
    outputs_ = {
        "pred_current": float(scores.loc["curr", "_pred"]),
        "pred_optimized": float(scores.loc["opt", "_pred"]),
    }
    penalty_ = {
        penalty_column: {
            "current": float(penalty_slack_table.loc["curr", penalty_column]),
            "suggested": float(penalty_slack_table.loc["opt", penalty_column]),
        }
        for penalty_column in penalty_slack_table.columns
        if "_penalty" in penalty_column
    }
    slack_ = {
        slack_column: {
            "current": float(penalty_slack_table.loc["curr", slack_column]),
            "suggested": float(penalty_slack_table.loc["opt", slack_column]),
        }
        for slack_column in penalty_slack_table.columns
        if "_slack" in slack_column
    }

    uplift_report_dict = {
        "run_id": row["run_id"].values[0],
        "timestamp": str(timestamp),
        "state": states_,
        "controls": controls_,
        "outputs": outputs_,
        "penalties": penalty_,
        "slack": slack_,
    }

    control_recommendations = []
    for control in on_controls:
        control_recommendations.append(
            {
                "tag_id": control,
                "run_id": row["run_id"].values[0],
                "value": scores.loc["opt", control],
            }
        )

    output_recommendation = {
        "run_id": row["run_id"].values[0],
        "tag_id": target,
        "baseline": float(scores.loc["curr", "_pred"]),
        "optimized": float(scores.loc["opt", "_pred"]),
    }
    return uplift_report_dict, control_recommendations, output_recommendation


def _optimize_dict(kwargs):
    return optimize(**kwargs)


def _get_target(td: TagDict) -> str:
    """ return target feature """
    target_list = td.select("model_target")
    return target_list[0]


def make_solver(params: dict, domain: List[Tuple]) -> Solver:
    """
    Creates an ask-tell solver from the tag dict

    Args:
        params: dict of pipeline parameters
        domain: List of domain bounds
    Returns:
        optimization solver object
    """
    solver_class = load_obj(params["solver"]["class"])
    solver_kwargs = params["solver"]["kwargs"]

    solver_kwargs.update({"domain": domain})

    return solver_class(**solver_kwargs)


def get_solver_bounds(
    input_data: pd.DataFrame, td: TagDict, features: List
) -> Mapping[str, List[Tuple]]:
    """
    Add more appropriate bounds to controls, applying max_deltas
    if available

    Args:
        current_value: DataFrame of current value for optimization,
        td: tag dictionary
        features: List of raw feature tags
    Returns:
        bounded solver instance
    """
    bounds_dict = {}
    controls = get_controls(features, td)
    for idx in input_data.index:
        current_value = input_data.loc[[idx], :]
        solver_bounds = []
        on_controls = get_on_features(current_value, td, controls)
        for control in on_controls:
            lower_bounds, upper_bounds = _get_single_control_bounds(
                control, current_value, td
            )
            solver_bounds.append((max(lower_bounds), min(upper_bounds)))
        bounds_dict[idx] = solver_bounds
    return bounds_dict


def _get_single_control_bounds(control, current_value, td):
    control_entry = td[control]
    op_min = control_entry["op_min"]
    op_max = control_entry["op_max"]
    lower_bounds = [op_min]
    upper_bounds = [op_max]
    if not pd.isna(control_entry["max_delta"]):
        current_val = current_value[control].iloc[0]
        if (current_val < op_min) or (current_val > op_max):
            logger.warning(
                f"Current Value for Control f{control} f{current_val} "
                f"is outside of range [f{op_min}, {op_max}]"
            )
        lower_bounds.append(current_val - control_entry["max_delta"])
        upper_bounds.append(current_val + control_entry["max_delta"])
    return lower_bounds, upper_bounds


def get_repairs(
    input_data: pd.DataFrame, td: TagDict, features: List
) -> Mapping[str, List[Callable]]:
    repairs_dict = {}
    controls = get_controls(features, td)
    controls_with_constraints = td.select("constraint_set", pd.notnull)
    for idx in input_data.index:
        current_value = input_data.loc[[idx], :]
        on_controls = get_on_features(current_value, td, controls)
        repairs_dict[idx] = [
            _make_set_repair(td, col)
            for col in (set(on_controls) & set(controls_with_constraints))
        ] or None
    return repairs_dict


def get_penalties(input_data: pd.DataFrame) -> Mapping[str, List[Callable]]:
    penalties_dict = {}
    for idx in input_data.index:
        penalties_dict[idx] = []
    return penalties_dict


def make_stopper(params: dict) -> Optional[BaseStopper]:
    """
    Creates a stopper using configured params

    Args:
        params: dict of pipeline parameters
    Returns:
        optimization stopper object
    """
    if params["stopper"]:
        stopper_class = load_obj(params["stopper"]["class"])
        stopper_kwargs = params["stopper"]["kwargs"]
        return stopper_class(**stopper_kwargs)
    return None


def _make_set_repair(td: TagDict, column: str) -> Repair:
    """ Creates a new constraint set repair for a given column """
    constraint_set = literal_eval(td[column]["constraint_set"])
    return repair(column, "in", constraint_set)


def get_feature_list(model, data):
    return model[0].transform(data.head()).columns.tolist()


def bulk_optimize(  # pylint:disable=too-many-locals
    params: dict,
    td: TagDict,
    data: pd.DataFrame,
    model: Any,
    penalty_dict: Mapping[str, List[penalty]],
    bounds_dict: Mapping[str, List[Tuple]],
    repairs_dict: Mapping[str, List[repair]],
    features: List[str],
    sense: str = "maximize",
) -> Dict:
    """
    Create recommendations for a whole dataframe in row by row.

    Args:
        params: dict of pipeline parameters
        td: tag dictionary
        data: dataframe to process
        model: model object.
        penalty_dict: A dictionary of penalties for each shift. The keys are row ids,
        the values are a list of user-defined penalties.
        bounds_dict: A dictionary of bounds to apply for each "problem" or shift.
        Keys are row ids, and values are Lists of tuples for each controllable variable
        repairs_dict: A dictionary of repairs to apply. The keys are shift_ids, and the
        values are Lists of callable repairs
        features: List of tag-level features that will form the basis of our model.
        sense: whether to maximize or minimize the objective.
    Returns:
        recommendations, recommended_controls, projected_optimization
    """
    # do not use parallel processing in the model
    # since we are parallelizing over rows in the dataframe
    try:
        model.set_params(estimator__n_jobs=1)
    except (ValueError, AttributeError):
        pass

    n_jobs = params["n_jobs"]

    controls = get_controls(features, td)
    target = params.get("target", "model_target")

    # for now, we show all model-states in the UI
    ui_states = [f for f in features if f not in controls]

    factory = StatefulOptimizationProblemFactory(
        model=model,
        data=data,
        penalty_dict=penalty_dict,
        bounds_dict=bounds_dict,
        repairs_dict=repairs_dict,
        features=features,
        td=td,
        sense=sense,
    )

    def yield_dicts():
        # we iterate over rows as single-row dataframes
        # instead of pd.Series in order to preserve dtypes
        for idx in data.index:
            row = data.loc[[idx], :]
            on_controls = get_on_features(row, td, controls)

            row_solver = make_solver(params, bounds_dict[idx]) if on_controls else None
            problem = factory.make_problem(idx)

            yield dict(
                timestamp=row.at[idx, "timestamp"],
                row=row,
                ui_states=ui_states,
                controls=controls,
                on_controls=on_controls,
                target=target,
                problem=deepcopy(problem),
                solver=row_solver,
                stopper=deepcopy(make_stopper(params)),
            )

    if n_jobs > 1:
        results = _parallelized_optimize(len(data), _optimize_dict, yield_dicts, n_jobs)
    else:
        results = tqdm(
            [_optimize_dict(kwargs) for kwargs in yield_dicts()], total=len(data)
        )
    uplift_results, control_results, output_results = list(zip(*results))
    uplift_results = pd.DataFrame(list(uplift_results))
    control_results = pd.DataFrame(list(list(cr for cr in control_results)))
    output_results = pd.DataFrame(list(output_results))

    return_latest_problem = params.get("return_problem", False)
    latest_problem = (
        factory.make_problem(data.index[-1])
        if return_latest_problem
        else return_latest_problem
    )

    return {
        "recommendations": uplift_results,
        "recommended_controls": control_results,
        "projected_optimization": output_results,
        "latest_problem": latest_problem,
    }


def bulk_optimize_context(  # pylint:disable=too-many-locals
    params: dict,
    td: TagDict,
    data: pd.DataFrame,
    model: Any,
    penalty_dict: Mapping[str, List[penalty]],
    bounds_dict: Mapping[str, List[Tuple]],
    repairs_dict: Mapping[str, List[repair]],
    features: List[str],
    sense: str = "maximize",
) -> Dict:
    """
    Create recommendations for a whole dataframe in row by row.
    The list of penalties must be defined by the user. As a default,
    there is an empty list placeholder.

    Args:
        params: dict of pipeline parameters
        td: tag dictionary
        data: dataframe to process
        model: model object.
        penalty_dict: A dictionary of penalties for each shift. The keys are row ids,
        the values are a list of user-defined penalties.
        bounds_dict: A dictionary of bounds to apply for each "problem" or shift.
        Keys are row ids, and values are Lists of tuples for each controllable variable
        repairs_dict: A dictionary of repairs to apply. The keys are shift_ids, and the
        values are Lists of callable repairs
        sense: whether to maximize or minimize the objective.
        features: List of tag-level features that will form the basis of our model.

    Returns:
        recommendations, recommended_controls, projected_optimization
    """
    # do not use parallel processing in the model
    # since we are parallelizing over rows in the dataframe
    try:
        model.set_params(estimator__n_jobs=1)
    except (ValueError, AttributeError):
        pass

    controls = get_controls(features, td)

    for feature in controls:
        if pd.isna(td[feature]["op_min"]) or pd.isna(td[feature]["op_max"]):
            raise ValueError(f"Operating Ranges for f{feature} must be specified.")

    # These params are specified in the model parameters file, not the
    # recommend params which are passed.
    window_size = get_reformatter_property(model, "length", default=1)
    stride = get_reformatter_property(model, "stride", default=1)

    ContextualOptimizationProblemFactory(
        model=model,
        data=data,
        penalty_dict=penalty_dict,
        bounds_dict=bounds_dict,
        repairs_dict=repairs_dict,
        features=features,
        td=td,
        sense=sense,
        window_size=window_size,
        objective_kwargs=params["objective_kwargs"],
    )

    # Sort values of index, then reset to operate on RangeIndex w/no gaps
    # the reset_Index call means we have to map back to original index with
    # data['index'][idx] for getting bounds, repairs and penalties.
    index_col = data.index.name or "index"  # when not set, index has no name
    data.sort_index(inplace=True)
    data.reset_index(inplace=True)

    def yield_dicts():
        # we iterate over rows as single-row dataframes
        # instead of pd.Series in order to preserve dtypes
        for idx in range(window_size - 1, data.shape[0], stride):
            # when window_size = 1, there is a context df with 0 rows
            row = data.iloc[[idx], :]
            on_controls = get_on_features(row, td, controls)

            if on_controls:
                # the normal case: we have at least one control variable
                # that we want to optimize
                row_solver_bounds = bounds_dict[data[index_col][idx]]
                row_solver = make_solver(params, row_solver_bounds)
                repairs = repairs_dict[data[index_col][idx]]
                penalty_list = penalty_dict[data[index_col][idx]]

            else:
                # if all machines are off, there is no recommendation to be
                # produced and we simply create a dummy problem
                row_solver = None
                repairs = None
                penalty_list = None

            problem = StatefulContextualOptimizationProblem(
                model,
                state=row,
                context_data=data.iloc[max(idx - window_size + 1, 0) : idx, :],
                optimizable_columns=on_controls,
                repairs=repairs,
                sense=sense,
                objective_kwargs=params["objective_kwargs"],
                penalties=penalty_list,
            )
            yield dict(
                timestamp=row.at[idx, "timestamp"],
                row=row,
                ui_states=[
                    f for f in features if f not in controls
                ],  # show all states in UI
                controls=controls,
                on_controls=on_controls,
                target=params.get("target", "model_target"),
                problem=deepcopy(problem),
                solver=row_solver,
                stopper=deepcopy(make_stopper(params)),
            )

    if params["n_jobs"] > 1:
        results = _parallelized_optimize(
            len(data), _optimize_dict, yield_dicts, params["n_jobs"]
        )
    else:
        results = tqdm(
            [_optimize_dict(kwargs) for kwargs in yield_dicts()], total=len(data)
        )

    uplift_results, control_results, output_results = list(zip(*results))
    uplift_results = pd.DataFrame(list(uplift_results))
    control_results = pd.DataFrame(list(list(cr for cr in control_results)))
    output_results = pd.DataFrame(list(output_results))
    return {
        "recommendations": uplift_results,
        "recommended_controls": control_results,
        "projected_optimization": output_results,
    }


def _parallelized_optimize(
    n_counterfactual, optimize_func, optimize_data, n_jobs=None, chunksize=5
):
    """
    Parallelize the optimization over a large set of counterfactuals.
    If the counterfactuals are long-running and Ctrl^C is entered (raising a
    KeyboardInterrupt error, the loop terminates and exits with whatever results
    have so far been calculated.
    Args:
        n_counterfactual: Number of counterfactuals to run
        optimize_func: function running optimization loop, to be applied to each row.
        optimize_data: Generator yielding args for `optimize`
        n_jobs: Number of parallel works over which to parallelize.
            By default max resources available.
        chunksize: Number of items to pass to a given worker. Keep workers busy.

    Returns:
        List of tuples, each holding 3 dictionaries of optimization output.
    """
    # we use imap_unordered (lazy pool.map) to gather results as they're returned.
    with Pool(n_jobs) as pool:
        results = [({}, {}, {})] * n_counterfactual
        try:
            pool_imap_data = pool.imap_unordered(
                optimize_func, optimize_data(), chunksize=chunksize
            )
            for idx, counterfactuals in tqdm(
                enumerate(pool_imap_data), total=n_counterfactual
            ):
                results[idx] = counterfactuals
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            pool.close()

        pool.close()
        pool.join()
    return results
