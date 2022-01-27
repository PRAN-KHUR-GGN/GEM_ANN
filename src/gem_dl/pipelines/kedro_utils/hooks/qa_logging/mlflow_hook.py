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
from contextlib import suppress

with suppress(ImportError):
    import mlflow
    import mlflow.sklearn as mlf_sk
    import mlflow.tensorflow as mlf_tf

import os
import sys
import tempfile
from typing import Any, Dict

import pandas as pd
import yaml
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from matplotlib.figure import Figure


class MLFlowHooks:
    """Namespace for grouping all model-tracking hooks with MLflow together.
    """

    def __init__(self):
        # log models overwrites if more than one model are generated.
        # Also, we are logging the model dataset versions anyways.
        with suppress(NameError):
            mlf_sk.autolog(log_models=False)
            mlf_tf.autolog(log_models=False)

    @hook_impl
    def before_pipeline_run(
        self,  # pylint:disable=unused-argument
        run_params: Dict[str, Any],
        pipeline: Pipeline,
    ):  # pylint:disable=unused-argument
        """Hook implementation to set the storage location for mlflow data"""

        mlflow.start_run(run_name=run_params["run_id"])
        mlflow.log_params(run_params)

    @hook_impl
    def after_node_run(
        self,
        node: Node,
        outputs: Dict[str, Any],
        inputs: Dict[str, Any],  # pylint:disable=unused-argument
    ) -> None:  # pylint:disable=unused-argument
        """Hook implementation to add model tracking after some node runs.
        In this example, we will:
        * Log the parameters after the data splitting node runs.
        * Log the model after the model training node runs.
        * Log the model's metrics after the model evaluating node runs.
        """

        """
        Hook will add an attribute 'catalog_entry` to all matplotlib figures that are
        returned as an output
        """
        # Find plots if returned, or returned as dict of plots
        def _extract_figures(dict_obj, namespace_str=node.namespace):
            return {
                f"plots_{namespace_str}_{k}": v
                for k, v in dict_obj.items()
                if isinstance(v, Figure)
            }

        plot_artifacts = _extract_figures(outputs)

        for k, v in outputs.items():
            # check if returning dict of figs as one output, then extract figures dict
            if isinstance(v, dict):
                plot_artifacts = {**plot_artifacts, **_extract_figures(v)}

            # Log model performance metrics
            if "metrics" in k:

                metrics_dict = outputs[k]
                # Log performance metrics -> convert to a dict if df is
                # returned with metric name on index
                if isinstance(outputs[k], pd.DataFrame):
                    metrics_dict = dict(outputs[k].to_records())

                metrics_dict = {
                    f"{node.short_name}.{k}": v for k, v in metrics_dict.items()
                }
                mlflow.log_metrics(metrics_dict)

        if plot_artifacts:
            with tempfile.TemporaryDirectory() as temp_dir:
                for k, v in plot_artifacts.items():
                    v.savefig(os.path.join(temp_dir, f"{k}"))
                    mlflow.log_artifacts(temp_dir)

    @hook_impl
    def after_pipeline_run(
        self,  # pylint:disable=unused-argument
        run_params: Dict[str, Any],
        run_result: Dict[str, Any],  # pylint:disable=unused-argument
        pipeline: Pipeline,
        catalog: DataCatalog,
    ) -> None:  # pylint:disable=unused-argument
        """Hook implementation to end the MLflow run
        after the Kedro pipeline finishes.
        """
        _log_kedro_info(run_params, pipeline, catalog)
        mlflow.end_run()


def _log_kedro_info(
    run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
) -> None:
    # this will have all the nested structures (duplicates)
    parameters = {
        input_param: catalog._data_sets[input_param].load()
        for input_param in pipeline.inputs()
        if "param" in input_param
    }
    # similar to context.params
    parameters.update(run_params.get("extra_params", {}))
    parameters_artifacts = {
        f"kedro_{_sanitise_kedro_param(param_name)}": param_value
        for param_name, param_value in parameters.items()
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(temp_dir + "/params.yml", "w") as f:
            yaml.dump(parameters_artifacts, f, default_flow_style=False)
            mlflow.log_artifact(temp_dir + "/params.yml")

    # Log tag dictionaries along with parameters once at end of pipeline run
    tag_dict_artifacts = {
        input_param: catalog._data_sets[input_param].load().to_frame()
        for input_param in pipeline.inputs()
        if "td" in input_param
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        for td_name, td in tag_dict_artifacts.items():
            td.to_csv(temp_dir + f"/{td_name}.csv", index=False)
            mlflow.log_artifact(temp_dir + f"/{td_name}.csv")
    mlflow.log_params(
        {
            "kedro_run_args": " ".join(
                repr(a) if " " in a else a for a in sys.argv[1:]
            ),
            "kedro_dataset_versions": list(_get_dataset_versions(catalog, pipeline)),
        }
    )


def _sanitise_kedro_param(param_name):
    sanitised_param_name = param_name.replace(":", "_")
    return sanitised_param_name


def _get_dataset_versions(catalog: DataCatalog, pipeline: Pipeline):
    for ds_name, ds in sorted(catalog._data_sets.items()):
        ds_in_out = ds_name in pipeline.all_outputs()
        try:
            save_ver = ds.resolve_save_version() if ds_in_out else None
            load_ver = ds.resolve_save_version() if ds_in_out else None
        except AttributeError:
            save_ver = None
            load_ver = None
        if save_ver or load_ver:
            version_info = {
                "name": ds_name,
                "save_version": save_ver,
                "load_version": load_ver,
            }
            yield version_info
