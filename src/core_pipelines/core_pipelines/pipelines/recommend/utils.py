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
Recommend pipeline utils.
"""
import abc
from typing import Any, Hashable, List, Mapping, Optional, Tuple, Union

import pandas as pd
from optimizer import repair, penalty
from optimizer.problem import (
    OptimizationProblem,
    StatefulOptimizationProblem,
    StatefulContextualOptimizationProblem,
)
from optimus_core.core.tag_management import TagDict


def check_problem_objects(
    data: pd.DataFrame,
    penalty_dict: Mapping[Hashable, List[penalty]],
    bounds_dict: Mapping[Hashable, List[Tuple]],
    repairs_dict: Mapping[Hashable, List[repair]],
    features: List[str],
):
    """Check that the objects needed to create problem objects are in the right format.

    Args:
        data: counterfactual data.
        penalty_dict: A dictionary of penalties for each shift. The keys are row ids,
        the values are a list of user-defined penalties.
        bounds_dict: A dictionary of bounds to apply for each "problem" or shift.
        Keys are row ids, and values are Lists of tuples for each controllable variable
        repairs_dict: A dictionary of repairs to apply. The keys are shift_ids, and the
        values are Lists of callable repairs
        features: List of tag-level features that will form the basis of our model.

    Raises:
        ValueError when provided objects are in the wrong format.
    """
    # validate dict format
    if (
        (len(penalty_dict) != len(data))
        or (len(bounds_dict) != len(data))
        or (len(repairs_dict) != len(data))
    ):
        raise ValueError(
            f"Something's missing: You've given a dataset with "
            f"{len(data)} rows, penalties for "
            f"{len(penalty_dict)} rows, bounds for {len(bounds_dict)} "
            f"rows, and repairs for {len(repairs_dict)} rows."
        )

    # validate features present in df
    if not all([feature in data.columns for feature in features]):
        raise ValueError(
            "Please make sure each of the raw features in `features` is"
            " in your input data"
        )


def get_controls(features: List, td: TagDict) -> List[str]:
    """Get controllable features.

    Args:
        features: list of all features.
        td: tag dictionary

    Returns:
        List of controllable feature names.
    """
    return [f for f in features if f in td.select("tag_type", "control")]


def get_on_features(
    current_value: pd.DataFrame, td: TagDict, controls: List[str]
) -> List[str]:
    """Determine which features are "on" using the tag dictionary.

    Args:
        current_value: single row DataFrame describing the current values of features.
        td: tag dictionary
        controls: controllable features used to determine on/off flag.

    Returns:
        List of strings, names of features that are "on".
    """
    on_controls = []
    for feature in controls:
        on_flag = all(
            [current_value[d].iloc[0] > 0.5 for d in td.dependencies(feature)]
        )
        if on_flag:
            on_controls.append(feature)
    return on_controls


def check_operating_ranges(td: TagDict, controls: List[str]):
    """Ensure that the operating range is specified for each control.

    Args:
        td: tag dictionary.
        controls: list of features that should have operating ranges.

    Raises:
        ValueError if an operating range is not specified.
    """
    for feature in controls:
        if pd.isna(td[feature]["op_min"]) or pd.isna(td[feature]["op_max"]):
            raise ValueError(f"Operating Ranges for f{feature} must be specified.")


class BaseProblemFactory(abc.ABC):
    """Base class to handle common factory operations.
    """

    def __init__(
        self,
        model: Any,
        data: pd.DataFrame,
        penalty_dict: Mapping[Hashable, List[penalty]],
        bounds_dict: Mapping[Hashable, List[Tuple]],
        repairs_dict: Mapping[Hashable, List[repair]],
        features: List[str],
        td: TagDict,
        sense: str,
    ):
        """Constructor.

        Args:
            model: model used as the objective in the OptimizationProblem.
            data: counterfactual data to construct the current state of the problem.
            penalty_dict: A dictionary of penalties for each shift. The keys are row
            ids, the values are a list of user-defined penalties.
            bounds_dict: A dictionary of bounds to apply for each "problem" or shift.
            Keys are row ids, and values are Lists of tuples for each control
            repairs_dict: A dictionary of repairs to apply. The keys are shift_ids, and
            the values are Lists of callable repairs
            features: List of tag-level features that will form the basis of our model.
            td: tag dictionary
            sense: str, "minimize" or "maximize".
        """
        check_problem_objects(data, penalty_dict, bounds_dict, repairs_dict, features)

        controls = get_controls(features, td)

        check_operating_ranges(td, controls)

        self.controls = controls
        self.model = model
        self.data = data
        self.penalty_dict = penalty_dict
        self.repairs_dict = repairs_dict
        self.features = features
        self.td = td
        self.sense = sense

    @abc.abstractmethod
    def make_problem(self, idx: Union[pd.Timestamp, int]) -> OptimizationProblem:
        """Make a problem object.

        Args:
            idx: Timestamp or integer, index of counterfactual to use.

        Returns:
            OptimizationProblem object for the index.
        """


class StatefulOptimizationProblemFactory(BaseProblemFactory):
    """Factory class for producing StatefulOptimizationProblems in the recommend
    pipeline.

    The goal of this class is to simplify the logic required to create a new
    optimization problem when running many counterfactuals.
    """

    def make_problem(
        self, idx: Union[pd.Timestamp, int]
    ) -> Union[StatefulOptimizationProblem, OptimizationProblem]:
        """Construct the problem object for the given row index.

        Args:
            idx: Timestamp or integer, index of counterfactual to use.

        Returns:
            OptimizationProblem object for the index.
        """
        row = self.data.loc[[idx], :]
        on_controls = get_on_features(row, self.td, self.controls)

        if on_controls:
            # The normal case: we have at least one control variable
            # that we want to optimize
            problem = StatefulOptimizationProblem(
                self.model,
                state=row,
                optimizable_columns=on_controls,
                repairs=self.repairs_dict[idx],
                penalties=self.penalty_dict[idx],
                sense=self.sense,
            )

        else:
            # if all machines are off, there is no recommendation to be
            # produced and we simply create a dummy problem
            problem = OptimizationProblem(self.model, sense=self.sense)

        return problem


def get_reformatter_property(
    model: Any, key: str, default: Optional[Any] = None
) -> Any:
    """Get a property from a model pipeline's data reformatter.

    Args:
        model: model pipeline to retrieve a property from.
        key: name of property to get the value for.
        default: returned if key not found in data reformatter.

    Returns:
        Any, the value of property.
    """
    try:
        value = (
            model.steps[-1][-1]
            .params.get("data_reformatter", {})
            .get("kwargs", {})
            .get(key, 1)
        )

    except AttributeError:
        value = default

    return value


class ContextualOptimizationProblemFactory(BaseProblemFactory):
    """Factory class for producing StatefulContextualOptimizationProblems in the
    recommend pipeline.

    The goal of this class is to simplify the logic required to create a new
    optimization problem when running many counterfactuals.
    """

    def __init__(  # pylint:disable=too-many-arguments
        self,
        model: Any,
        data: pd.DataFrame,
        penalty_dict: Mapping[Hashable, List[penalty]],
        bounds_dict: Mapping[Hashable, List[Tuple]],
        repairs_dict: Mapping[Hashable, List[repair]],
        features: List[str],
        td: TagDict,
        sense: str,
        window_size: int,
        objective_kwargs: dict,
    ):
        """Constructor.

        Args:
            model: model used as the objective in the OptimizationProblem.
            data: counterfactual data to construct the current state of the problem.
            penalty_dict: A dictionary of penalties for each shift. The keys are row
            ids, the values are a list of user-defined penalties.
            bounds_dict: A dictionary of bounds to apply for each "problem" or shift.
            Keys are row ids, and values are Lists of tuples for each control
            repairs_dict: A dictionary of repairs to apply. The keys are shift_ids, and
            the values are Lists of callable repairs
            features: List of tag-level features that will form the basis of our model.
            td: tag dictionary
            sense: str, "minimize" or "maximize".
            window_size: int, window size of prediction.
            objective_kwargs: keyword arguments passed to objective.
        """
        super().__init__(
            model=model,
            data=data,
            penalty_dict=penalty_dict,
            bounds_dict=bounds_dict,
            repairs_dict=repairs_dict,
            features=features,
            td=td,
            sense=sense,
        )

        sorted_index = data.index.sort_values()
        self.sorted_position = {value: i for i, value in enumerate(sorted_index)}
        self.position_to_index = dict(enumerate(sorted_index))
        self.window_size = window_size
        self.objective_kwargs = objective_kwargs

    def make_problem(
        self, idx: Union[pd.Timestamp, int]
    ) -> StatefulContextualOptimizationProblem:
        """Construct the problem object for the given row index.

        Args:
            idx: Timestamp or integer, index of counterfactual to use.

        Returns:
            OptimizationProblem object for the index.
        """
        row = self.data.loc[[idx], :]

        on_controls = get_on_features(row, self.td, self.controls)

        if on_controls:
            # the normal case: we have at least one control variable
            # that we want to optimize
            repairs = self.repairs_dict[idx]
            penalty_list = self.penalty_dict[idx]

        else:
            # if all machines are off, there is no recommendation to be
            # produced and we simply create a dummy problem
            repairs = None
            penalty_list = None

        # Do some juggling to avoid inferring the type and/or frequency of the index.
        idx_position = self.sorted_position[idx]
        previous_position = max(idx_position - self.window_size + 1, 0)
        context_indices = [
            self.position_to_index[i]
            for i in range(previous_position, idx_position + 1)
        ]

        return StatefulContextualOptimizationProblem(
            self.model,
            state=row,
            context_data=self.data.loc[context_indices, :],
            optimizable_columns=on_controls,
            repairs=repairs,
            penalties=penalty_list,
            sense=self.sense,
            objective_kwargs=self.objective_kwargs,
        )
