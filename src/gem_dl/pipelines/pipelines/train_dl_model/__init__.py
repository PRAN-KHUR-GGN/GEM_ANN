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
__version__ = "0.1.0"

from .keras_estimator import SamplingModelMixin, wrap_class
from .pipeline import create_pipeline

__all__ = [
    "wrap_class",
    "SamplingModelMixin",
    "create_pipeline",
]
