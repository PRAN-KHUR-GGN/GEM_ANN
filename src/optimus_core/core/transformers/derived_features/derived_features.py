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
Transformer of eng features
"""

from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from pydantic.main import ModelMetaclass
from sklearn.utils.validation import check_is_fitted

from ..base import Transformer
from .pydantic_models import DerivedFeaturesCookBook


class NotDagError(Exception):
    """Raised when graph is expected to be a dag but is not"""


class FunctionReturnError(Exception):
    """Raise when a function to create an eng feature doesnt create it"""


class DAGHandlerMixin:
    def __init__(self, config: Dict[str, Any], pydantic_model: ModelMetaclass):
        """Mixin that handles parsing `config` according to `pydantic_model`,
        initializating `self._graph`, and providing a method to create a derived feature
        corresponding to a given node.

        Args:
            config (Dict[str, Any]): especification of eng features, constraints
             and models
            pydantic_model (ModelMetaclass): pydantic model used to parse `config`
        """
        self._graph = self._get_graph(pydantic_model(cookbook=config))

    def _get_graph(self, cookbook: DerivedFeaturesCookBook) -> nx.DiGraph:
        graph = nx.DiGraph()
        for derived_feature, recipe in cookbook.cookbook.items():
            for dependency in recipe.dependencies:
                graph.add_edge(derived_feature, dependency)
            graph.nodes[derived_feature]["recipe"] = recipe
        if not nx.is_directed_acyclic_graph(graph):
            raise NotDagError("The graph of eng features is not a dag")
        return graph

    def _evaluate_node(self, x: pd.DataFrame, node_name: str) -> pd.DataFrame:
        recipe = self._graph.nodes[node_name]["recipe"]
        if hasattr(recipe.function, "predict"):
            ans = recipe.function.predict(x)
        elif callable(recipe.function):
            ans = recipe.function(x, recipe.dependencies, *recipe.args, **recipe.kwargs)
        else:
            raise AttributeError(
                f"Recipe function {recipe.function.__name__} should be callable or have a predict method"
            )
        if isinstance(ans, (pd.Series, np.ndarray)):
            x[node_name] = ans
        else:
            x = ans
        return x


class DerivedFeaturesMaker(Transformer, DAGHandlerMixin):
    def __init__(
        self,
        config: Dict[str, Any],
        pydantic_model: ModelMetaclass = DerivedFeaturesCookBook,
    ):
        """
        Creates the eng features especified by `config`. `config` must
        have the schema of `DerivedFeaturesCookBook`.

        Args:
            config (Dict[str, Any]): especification of eng features.

        Raises:
            NotDagError: if graph of eng features is not a dag.
        """
        DAGHandlerMixin.__init__(self, config, pydantic_model)

    def _not_leaf(self, node_name) -> bool:
        return self._graph.out_degree(node_name) > 0

    def fit(self, x: pd.DataFrame, y=None, **fit_params):
        """
        Checks the following:
        - if given input is a pandas DataFrame.
        - if leaf nodes of `graph` of eng features are present in `x`.
        - if functions associated with each eng feature actually create the feature.


        Args:
            x (pd.DataFrame): training data
            y: training y (no effect). Defaults to None.

        Raises:
            ValueError: if leaf node is not in `x`
            FunctionReturnError: if the function associated
             with an eng feature doesnt create it

        Returns:
            self
        """
        self.check_x(x)
        sample = x.head(2)
        filtered_toposort = []
        full_toposort = list(reversed(list(nx.topological_sort(self._graph))))
        for node_name in full_toposort:
            if self._not_leaf(node_name):
                filtered_toposort.append(node_name)
                before_derived = sample.copy()
                sample = self._evaluate_node(sample, node_name)
                self._validate_ans(sample, before_derived, node_name)
            elif node_name not in x:
                raise ValueError(
                    f"The following node must be present in x: {node_name}"
                )
        self.topological_sort_: List[str] = filtered_toposort
        return self

    def _validate_ans(
        self, ans: pd.DataFrame, sample: pd.DataFrame, node_name: str
    ) -> None:
        if not (
            isinstance(ans, pd.DataFrame)
            and set(ans.columns.tolist()) == set(sample.columns.tolist() + [node_name])
        ):
            raise FunctionReturnError(
                f"The function of eng feature {node_name} does not create this feature"
            )

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Creates derived features according to `graph` cookbook

        Args:
            x (pd.DataFrame): data to create eng features

        Returns:
            pd.DataFrame: original data with eng features.
        """
        check_is_fitted(self, "topological_sort_")
        for node_name in self.topological_sort_:
            x = self._evaluate_node(x, node_name)
        return x
