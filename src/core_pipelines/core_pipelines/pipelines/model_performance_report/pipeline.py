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
Model Training Pipeline
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_performance_report, generate_performance_report_figures


def create_pipeline():
    """
    Using this pipeline expects users to have a parameters file that includes
    a block like ::

        model_performance_report:
            output_path: .
            report_name: performance_report
            target: *train_model_target
            timestamp: True
            title: "Model Performance Report"
            author: "OptimusAI"
            subject: "OptimusAI Performance Report"
            shap:
                explainer: shap.KernelExplainer
                kwargs: {}


    Returns:

    """
    return pipeline(
        pipe=Pipeline(
            [
                node(
                    generate_performance_report_figures,
                    dict(
                        params="params:model_performance_report",
                        feature_importance="train_set_feature_importance",
                        train_set_predictions="train_set_predictions",
                        test_set_predictions="test_set_predictions",
                        model="model",
                        test_set_metrics="test_set_metrics",
                        train_set_metrics="train_set_metrics",
                        static_features="input",
                    ),
                    "report_figures",
                    name="generate_performance_report_figures",
                ),
                node(
                    generate_performance_report,
                    dict(
                        params="params:model_performance_report",
                        figures_dict="report_figures",
                    ),
                    None,
                    name="generate_performance_report",
                ),
            ]
        )
    )
