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

from kedro.pipeline import Pipeline, node, pipeline

from core_pipelines.core_pipelines.pipelines.train_model.nodes import (
    add_transformers,
    create_predictions,
    load_estimator,
    retrain_model,
    train_model,
)
from core_pipelines.core_pipelines.pipelines.train_model.reports import (
    generate_performance_report, 
    generate_performance_report_figures
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # Training model
            # ----------------------------------------------------------#
            # TRAIN MODEL PIPELINE
            # ----------------------------------------------------------#
            node(
                load_estimator,
                dict(params="params:lgbm_train_model"),
                "estimator",
                name="load_estimator",
            ),
            node(
                add_transformers,
                dict(td="td", estimator="estimator", params="params:lgbm_train_model"),
                "estimator_pipeline",
                name="add_transformers",
            ),
            node(
                train_model,
                dict(
                    params="params:lgbm_train_model",
                    data="train_set",
                    model="estimator_pipeline",
                ),
                dict(
                    model="train_set_model",
                    cv_results="train_set_cv_results",
                    feature_importance="train_set_feature_importance",
                ),
                name="train_model",
            ),
            node(
                create_predictions,
                dict(
                    params="params:lgbm_train_model",
                    data="train_set",
                    model="train_set_model",
                ),
                dict(
                    predictions="train_set_predictions", metrics="train_set_metrics"
                ),
                name="create_train_predictions",
            ),
            node(
                create_predictions,
                dict(
                    params="params:lgbm_train_model",
                    data="test_set",
                    model="train_set_model",
                ),
                dict(
                    predictions="test_set_predictions", metrics="test_set_metrics"
                ),
                name="create_test_predictions",
            ),
            node(
                retrain_model,
                dict(
                    params="params:lgbm_train_model",
                    model="train_set_model",
                    data="train_set",
                ),
                "model",
                name="retrain_model",
            ),
            node(
                generate_performance_report_figures,
                dict(
                    params="params:lgbm_train_model.report",
                    feature_importance="train_set_feature_importance",
                    train_set_predictions="train_set_predictions",
                    test_set_predictions="test_set_predictions",
                    model="model",
                    test_set_metrics="test_set_metrics",
                    train_set_metrics="train_set_metrics",
                    static_features="train_set",
                ),
                "report_figures",
                name="generate_performance_report_figures",
            ),
            node(
                generate_performance_report,
                dict(
                    params="params:lgbm_train_model.report",
                    figures_dict="report_figures",
                ),
                None,
                name="generate_performance_report",
            ),

        ]
    )
