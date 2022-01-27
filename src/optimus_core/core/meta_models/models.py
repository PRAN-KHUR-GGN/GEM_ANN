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

"""This module contains a collection of different meta-models available."""
from typing import Union

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array


class StackedModel(BaseEstimator, RegressorMixin):
    """Allows training stacked models."""

    def __init__(
        self, target_model: BaseEstimator, residual_model: BaseEstimator,
    ):
        """Creates an object with two models to fit on residual of first.

        Target model trains on the target column and the residual model trains on the
        residuals.

        Args:
            target_model: Sklearn model
            residual_model: Sklearn model
        """
        self.target_model = target_model
        self.residual_model = residual_model
        self.pred_main = []
        self.pred_residual = []
        self.is_fitted_ = False

    def fit(self, data: pd.DataFrame, target: pd.DataFrame):
        """Fits the training model.
        Args:
            data: training dataset.
            target: labels to train model on.
        Returns:
            instance of model.
        """
        _data, _target = check_X_y(data, target)
        self.target_model.fit(data, target)
        residual = target - self.target_model.predict(data)
        self.residual_model.fit(data, residual)

        # sklearn is_fitted function requires this to be set true
        self.is_fitted_ = True

        return self

    def predict(self, data: Union[list, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict using the trained model on ``data``.
        Args:
            data: DataFrame to do predictions on.

        Returns:
            DataFrame of predictions.
        """
        check_is_fitted(self)
        data = check_array(data)

        self.pred_main = self.target_model.predict(data)
        self.pred_residual = self.residual_model.predict(data)

        return self.pred_main + self.pred_residual
