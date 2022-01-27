# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
from functools import partial
from typing import Callable

from kedro.pipeline import Pipeline, node

from ..data_processing.nodes import *
from ..split.nodes import *
from ..pipelines.train_dl_model.nodes import *
from ..pipelines.train_dl_model.reports import generate_report_pages
from ..pipelines.train_model.nodes import create_predictions


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # Loading data
            node(
                preprocess_energy,
                ["energy"],
                "preprocessed_energy"
            ),
            node(
                func=split_data,
                inputs=dict(params="params:split", data="data"),
                outputs=dict(train="train_data", test="test_data"),
                name="split_data",
            ),
            node(
                    partial(create_keras_trainer, config_mapper=config_mapper),
                    inputs=dict(
                        td="td", data="train_data", params="params:train_deeplearning",
                    ),
                    outputs="keras_trainer_func",
                    name="create_keras_trainable",
            ),
            node(
                    prepare_tune_params,
                    inputs="params:train_deeplearning.tune_params",
                    outputs="tune_params",
                    name="prepare_params_for_ray_tune",
            ),
            node(
                    tune_keras_estimator,
                    dict(keras_trainer="keras_trainer_func", tune_params="tune_params"),
                    dict(best_config="best_config", tune_results="tune_results"),
                    name="tune_keras_estimator",
            ),
            node(
                    partial(create_best_model, config_mapper=config_mapper),
                    inputs=dict(
                        best_config="best_config",
                        data="train_data",
                        params="params:train_deeplearning",
                        td="td",
                    ),
                    outputs="model",
                    name="create_best_model",
                ),
            node(
                    fit_best_keras_model,
                    dict(
                        params="params:train_deeplearning",
                        data="train_data",
                        model="model",
                    ),
                    dict(
                        model="ts_model",
                        feature_importance="train_set_feature_importance",
                    ),
                    name="train_model_on_train_set",
            ),
            node(
                    create_predictions,
                    dict(
                        params="params:train_deeplearning",
                        data="test_data",
                        model="ts_model",
                    ),
                    dict(
                        predictions="test_set_predictions", metrics="test_set_metrics"
                    ),
                    name="create_test_predictions",
            ),
            node(
                    create_predictions,
                    dict(
                        params="params:train_deeplearning",
                        data="train_data",
                        model="ts_model",
                    ),
                    dict(
                        predictions="train_set_predictions", metrics="train_set_metrics"
                    ),
                    name="create_train_predictions",
            ),
            node(
                    fit_best_keras_model,
                    dict(
                        params="params:train_deeplearning",
                        data="data",
                        model="model",
                        wait_on="test_set_metrics",
                    ),
                    dict(
                        model="full_set_model",
                        feature_importance="full_feature_importance",
                    ),
                    name="fit_optimal_model_full_data",
            ),
            node(
                    generate_report_pages,
                    dict(
                        params="params:train_deeplearning.report",
                        feature_importance="train_set_feature_importance",
                        train_set_predictions="train_set_predictions",
                        test_set_predictions="test_set_predictions",
                        model="full_set_model",
                        train_set_metrics="train_set_metrics",
                    ),
                    "report_figures",
                    name="generate_performance_report_figures",
                ),


        ]
    )
