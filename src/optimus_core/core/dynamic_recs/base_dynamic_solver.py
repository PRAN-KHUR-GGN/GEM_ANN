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
"""Base Dynamic Solver."""

import abc
from typing import Dict, List, Tuple, Union

Value = Union[float, str]


class DynamicSolver(abc.ABC):
    """Base class for Dynamic Solvers."""

    @abc.abstractmethod
    def update_recs(
        self, rejected_values: Dict[str, Value]
    ) -> Dict[str, Tuple[Value, List[Value]]]:
        """Update the recs based on the rejected values.

        Args:
            rejected_values: Dict of rejected values.

        Returns:
            Dict of recs and their possible values.
        """
        if not _check_type(rejected_values):
            raise ValueError(
                "rejected_values must be of type Dict[str, Union[float, str]]"
            )


def _check_type(rejected_values: Dict[str, Union[float, str]]) -> bool:
    if not isinstance(rejected_values, dict):
        return False
    right_values = all(
        [
            isinstance(value, str) or isinstance(value, float) or isinstance(value, int)
            for value in rejected_values.values()
        ]
    )
    right_keys = all([isinstance(key, str) for key in rejected_values])
    return right_keys and right_values
