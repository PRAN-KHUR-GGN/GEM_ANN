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
"""Schemas required by pydantic to validate the format of the configuration"""

from typing import Any, Callable, Dict, Iterable, Mapping

import numpy as np
from optimus_core.core.utils import load_obj
from pydantic import BaseModel, conlist, validator


class DerivedFeaturesRecipe(BaseModel):
    """Holder of necessary info for creating an eng feature"""

    dependencies: conlist(str, min_items=1)
    function: Callable
    args: Iterable[Any] = []
    kwargs: Mapping[str, Any] = {}

    @validator("function", pre=True)
    def load_func(cls, func):
        """if `func` is a path to an object, load it."""
        if isinstance(func, str):
            return load_obj(func)
        return func

    class Config:
        arbitrary_types_allowed = True


class DerivedFeaturesCookBook(BaseModel):
    """Holder of info to create all eng features"""

    cookbook: Dict[str, DerivedFeaturesRecipe]


class GridSearchRecipe(DerivedFeaturesRecipe):
    """Extends `DerivedFeaturesRecipe` to support constraints"""

    lower_bound: float = -np.inf
    upper_bound: float = np.inf


class GridSearchCookBook(BaseModel):
    """Holder of info to create grid for grid search"""

    cookbook: Dict[str, GridSearchRecipe]
