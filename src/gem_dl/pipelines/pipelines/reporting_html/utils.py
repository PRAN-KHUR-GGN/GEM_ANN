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
notebook utils
"""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import pandas as pd
from IPython.display import Markdown, display
from kedro.framework.context import KedroContext
from kedro.framework.context import load_context as load_kedro_context
from optimus_core.core.tag_management import TagDict


@contextmanager
def set_env_var(key, value):
    """
    Simple context manager to temporarily set an env var.
    """
    current = os.environ.get(key)
    try:
        os.environ[key] = value
        yield
    finally:
        if current is None:
            del os.environ[key]
        else:
            os.environ[key] = current


def load_context(start_path=None, max_depth=4, env=None, **kwargs) -> KedroContext:
    """
    Tries to load the kedro context from a notebook of unknown location.
     Assumes that the notebook is placed somewhere in the kedro project.

    Args:
        start_path: starting point for the search. We try to load the
         context from here and continue up the chain of parents
        max_depth: max number of parents to visit
        env: kedro environment to use. Defaults to `KEDRO_ENV`
         environment variable if available
        **kwargs: kwargs for `kedro.context.load_context`

    """
    start_path = Path(start_path) if start_path is not None else Path.cwd()
    parents = list(start_path.parents)
    to_check = [start_path] + parents[:max_depth]

    env = env or os.environ.get("KEDRO_ENV")

    for path in to_check:
        if path.joinpath(".kedro.yml").exists():
            return load_kedro_context(path, env=env, **kwargs)

    raise RuntimeError("Kedro context not found.")


def mprint(text: str, **kwargs):
    """
    Renders Markdown text in a notebook.
    Args:
        text: raw Markdown
        **kwargs: arguments for Ipython.display.display
    """
    display(Markdown(text), **kwargs)


def filter_df_cols(data: pd.DataFrame, td: TagDict, params: Dict):
    """
    Return the input dataframe (e.g. the master table) subset only for model features
    and target variable
    Args:
        params: parameters to be used
        data: dataset to be processed
        td: tag dictionary

    Returns:
        reduced dataset
    """
    model_vars = get_model_vars(params, td)
    return {"data_filtered": data[model_vars]}


def get_model_vars(params, td):
    """
    Get model features and target variable
    Args:
        params: parameters to be used
        td: tag dictionary

    Returns:
        list containing model inputs & output
    """
    target_col = params.get("target", "model_target")
    # target variables is NOT  explicitly specified in params
    if target_col == "model_target":
        target_col = td.select(target_col)

    feat_cols = params.get("features", "model_feature")
    # model features are NOT EXPLICITLY specified in params
    if isinstance(feat_cols, str):
        feat_cols = td.select(feat_cols)

    # cast to list to allow list concatenation
    if not isinstance(target_col, list):
        target_col = [target_col]
    model_vars = feat_cols + target_col

    return model_vars
