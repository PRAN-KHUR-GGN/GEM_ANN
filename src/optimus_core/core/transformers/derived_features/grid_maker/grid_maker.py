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
#

from functools import reduce
from itertools import combinations
from typing import Any, Dict, List, Set

import networkx as nx
import pandas as pd
import toolz
from pydantic.main import ModelMetaclass
from toposort import toposort

from ..derived_features import DAGHandlerMixin, DerivedFeaturesMaker
from ..pydantic_models import GridSearchCookBook


class GridMaker(DAGHandlerMixin):
    def __init__(
        self,
        config: Dict[str, Any],
        current_shift: pd.DataFrame,
        allowed_values: Dict[str, List[float]],
        pydantic_model: ModelMetaclass = GridSearchCookBook,
    ):
        """Grid search implementation based on the DAG of engineered features,
         constraints and models.

        Args:
            config (Dict[str, Any]): especification of eng features, constraints
             and models
            allowed_values (Dict[str, List[float]]): mapping from tags to its
             possible values
            pydantic_model (ModelMetaclass, optional): pydantic model to parse `config`
            Defaults to GridSearchCookBook.
        """
        config = self._add_current_shift_to_config(config, current_shift)
        super().__init__(config, pydantic_model)
        self._allowed_values = {
            tag: self._values_to_df_with_dummy(values, tag)
            for tag, values in allowed_values.items()
        }
        self._initialize_current_shift(config, pydantic_model, current_shift)
        self._toposort: List[Set[str]] = None
        self.target: str = None

    def _initialize_current_shift(self, config, pydantic_model, current_shift):
        config_with_no_current_shift = {
            feature: recipe_config
            for feature, recipe_config in config.items()
            if not self._needs_current_shift(recipe_config)
        }
        self._current_shift = DerivedFeaturesMaker(
            config_with_no_current_shift, pydantic_model
        ).fit_transform(current_shift)

    def _needs_current_shift(self, recipe_config):
        return "current_shift" in recipe_config.get("kwargs", {})

    def _values_to_df_with_dummy(self, values, tag, dummy_col: str = "dummy"):
        df = pd.Series(values, name=tag).to_frame()
        df[dummy_col] = 1
        return df

    def optimize(
        self, dummy_col: str = "dummy", sense: str = "maximize", n_top: int = 1
    ) -> pd.DataFrame:
        """Get the `n_top` maximum/minimum of highest node in dag
        (should be objective function) depending on whether `sense` is
        `maximize` or something else.

        Args:
            sense (str, optional): Whether to maximize/minimize. Defaults to "maximize".
            n_top (int, optional): Number of best solutions. Defaults to 1.

        Returns:
            pd.DataFrame: `n_top` rows in mode `sense`.
        """
        self._setup()
        node_name = None
        for level in self._toposort[1:]:
            while level:
                node_name = level.pop()
                self._prepare_node_inputs(node_name)
                self._evaluate_node(node_name)
        self.target = node_name
        grid = self._allowed_values[node_name].drop(columns=[dummy_col])
        sense_method = "nlargest" if sense == "maximize" else "nsmallest"
        return getattr(grid, sense_method)(n_top, self.target)

    def _prepare_node_inputs(self, node_name: str):
        successors = set(self._graph.successors(node_name))
        successors_dfs = sorted(
            [
                self._allowed_values[succ]
                for succ in successors
                if succ in self._allowed_values
            ],
            key=lambda df: df.shape[1],
            reverse=True,
        )
        node_inputs = reduce(pd.merge, successors_dfs)
        missing = successors - set(node_inputs.columns)
        if missing:
            node_inputs = self._set_values_current_shift(node_inputs, missing)
        self._allowed_values[node_name] = node_inputs

    def _set_values_current_shift(
        self, df: pd.DataFrame, missing: Set[str]
    ) -> pd.DataFrame:
        while missing:
            col = missing.pop()
            df[col] = self._current_shift[col]
        return df

    def _evaluate_node(self, node_name: str):  # pylint: disable=arguments-differ
        res = super(GridMaker, self)._evaluate_node(
            self._allowed_values[node_name], node_name
        )
        bounds = (
            self._graph.nodes[node_name]["recipe"].lower_bound,
            self._graph.nodes[node_name]["recipe"].upper_bound,
        )
        self._allowed_values[node_name] = res[res[node_name].between(*bounds)]

    def _setup(self):
        self._graph = _DagTransformer().transform(self._graph)
        self._toposort = list(
            toposort(toolz.valmap(set, nx.to_dict_of_lists(self._graph)))
        )

    def _add_current_shift_to_config(self, config, current_shift):
        for feature, recipe_config in config.items():
            if recipe_config.get("kwargs", {}).get("current_shift", False):
                config[feature]["kwargs"]["current_shift"] = current_shift
        return config


class _DagTransformer:
    """Group of methods to transform a dag of feature dependence
    into a dag of sequence of operations.
    """

    def transform(self, graph):
        """adds edges from `a` to `b` whenever the descendants of `b`
        are a subset of those of `a`. When this happens, we also remove
        any existing edge between `a` and the descendants of `b`.

        Args:
            graph ([type]): dag

        Returns:
            [type]: transformed dag
        """
        for node1, node2 in combinations(graph.nodes, 2):
            self._compare_nodes(graph, node1, node2)
        return graph

    def _prune_inheritance(self, graph, parent, child, grandchildren):
        graph.add_edge(parent, child)
        for grandchild in grandchildren:
            if graph.has_edge(parent, grandchild):
                graph.remove_edge(parent, grandchild)

    def _compare_nodes(self, graph, node1, node2):
        children1 = nx.descendants(graph, node1)
        children2 = nx.descendants(graph, node2)
        if len(children2) == 0 or len(children1) == 0:
            return
        intersection = children1 & children2
        if children1.issubset(children2):
            self._prune_inheritance(graph, node2, node1, intersection)
            return
        if children2.issubset(children1):
            self._prune_inheritance(graph, node1, node2, intersection)
            return
