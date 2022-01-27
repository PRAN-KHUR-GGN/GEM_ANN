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
import logging
from typing import Any, Dict, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
from optimus_core.core import utils
from tensorflow.python.keras.wrappers.scikit_learn import BaseWrapper


def wrap_class(
    base_class: BaseWrapper,
    data_reformatter: Union[None, Dict[str, Any]] = None,
    probabilistic: bool = False,
) -> BaseWrapper:
    """
    Depending on the values of `data_reformatter` and `probabilistic`, extra
    functionality is added to `base_class` (no added func by default).
    If `data_reformatter` is a dictionary containing the class path and kwargs
    of a data_reformatter, e.g., `tf.keras.preprocessing.sequence.TimeseriesGenerator`
    , a mixin calling this reformatter before `fit` and `predict` is added to
    `base_class`.
    If `probabilistic` is set to `True`, then the `SamplingModelMixin` is added.

    Args:
        base_class (BaseWrapper): base estimator class. Tipically either
            `KerasRegressor` or `KerasClassifier`.
        data_reformatter (Union[None, Dict[str, Any]], optional): Data reformater
            especification, e.g., `{"class": "...TimeseriesGenerator", "kwargs":"..."}`.
            Defaults to None.
        probabilistic (bool, optional): wether to add `SamplingModelMixin` or not.
            Defaults to False.

    Returns:
        BaseWrapper: `base_class` with the appropiate extensions.
    """

    if data_reformatter is None and not probabilistic:
        return base_class

    base_classes = []

    if data_reformatter is not None:
        data_reformatter_kwargs = data_reformatter.get("kwargs", {})
        data_reformatter = utils.load_obj(data_reformatter["class"])
        base_classes.append(
            _get_reformatter_mixin(data_reformatter, data_reformatter_kwargs)
        )

    base_classes.append(base_class)

    if probabilistic:
        base_classes.append(SamplingModelMixin)

    class KerasEstimator(*base_classes):
        pass

    return KerasEstimator


def _get_reformatter_mixin(data_reformatter, data_reformatter_kwargs):
    class ReformatterMixin:
        """
        Extension of a keras estimator, that reformats the data according to the given
        `data_reformatter` prior to calling `fit` or `predict`.
        """

        def _reformat_xy(self, x, y):
            # If input shape > 1 (probably need to swap this out
            if isinstance(x, pd.DataFrame):
                x = x.values
                y = y.values
            if data_reformatter is not None:
                data = data_reformatter(x, y, **data_reformatter_kwargs)
                x = np.vstack([arr_[0] for arr_ in data])
                y = np.vstack([arr_[1][:, np.newaxis] for arr_ in data]).squeeze()

            return x, y

        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            validation_data: Tuple[np.ndarray, np.ndarray] = None,
            **kwargs,
        ):
            """
            Fit the keras model, then save to disk
            Args:
                X: Numpy array of input features/data
                y: Numpy array of targets.
                validation_data: Tuple of np.arrays, of same dimensions as X, y.
                To be used for evaluation.
                is_tuning: Flag to determine if model should be saved to disk
                by default.

            Returns: An instance of self.

            """

            X, y = self._reformat_xy(X, y)
            if validation_data is not None:
                val_X, val_y = self._reformat_xy(validation_data[0], validation_data[1])
                validation_data = (val_X, val_y)
            return super().fit(X, y, validation_data=validation_data, **kwargs)

        def predict(self, x: pd.DataFrame, reshape=False, window_size=None, **kwargs):
            """
            Enable model.predict, for usage within the optimizer ask/tell framework.

            Args:
                x: Input data.
                reshape: When true, "hard" reshapes data with numpy, rather than
                converting to a time-series tensor with `self._reformat_xy`.
                window_size: The number of timesteps needed to create a row to predict
                on.

            Returns:
                A numpy array of predicted values.

            """

            # This is an edge case, where when predicting we pass data that's a
            # multiple of min seq length, we just can re-arrange the data accordingly.
            if reshape:
                if window_size is None:
                    window_size = data_reformatter_kwargs.get("length", 1)
                n_rows = x.shape[0] // window_size
                if n_rows * window_size != x.shape[0]:
                    logging.warning(
                        "You're trying to reshape particles, "
                        f"expecting x to have {window_size*n_rows} but got {x.shape[0]}"
                    )
                x = np.reshape(x.values, newshape=(n_rows, window_size, x.shape[1]))
            else:
                x, _ = self._reformat_xy(x, x)
            return super().predict(x, **kwargs)

    return ReformatterMixin


class SamplingModelMixin:
    """
    Extension of a keras estimator, that includes methods for sampling from the
    model being build, and returns numpy arrays.
    """

    @staticmethod
    def _check_X(X: Union[pd.DataFrame, np.ndarray]):
        """Convert pandas objects to np arrays.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data

        Returns:
            np.ndarray: converted data.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        return X

    def predict_dist(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Return the raw tensor of the model object. This is the distribution layers
        object.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input dataset

        Returns:
            [type]: A distribution (if final layer is a distribution.)
        """
        X = self._check_X(X)
        return self.model(X)

    def sample(
        self, X: Union[pd.DataFrame, np.ndarray], n_samples: Union[int, Tuple[int]]
    ):
        """
        Sample from the model.
        Here d is the dimension of the output

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data/features
            n_samples Union[int,Tuple[int]]: Size of samples to generate.

        Returns:
            np.ndarray: Samples from model. For model that predicts d dimensional
            outcome, and an input array of n rows, output has dimension
            (n_samples x n x d array)
        """
        X = self._check_X(X)
        return_tensor = self.model(X)
        if hasattr(return_tensor, "sample"):
            # Sample aleatoric uncertainty.
            preds = return_tensor.sample(n_samples).numpy()
        else:
            # Sample epistemic uncertainty via the model
            # For models with dropout, we'll want training=True for stochasticity
            preds = np.hstack(
                [self.model(X, training=True).numpy() for _ in range(n_samples)]
            ).T
        return preds

    def predict_point(
        self, X: Union[pd.DataFrame, np.ndarray], n_samples: Union[int, Tuple[int]]
    ):
        """
        Convenience function for point predictions with rough uncertainty. Under the
        hood, this calls sample, then aggregates along sample dimension and returns
        a tuple of np.narrays, representing the mean and std of the sampled values.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Array of input values
            n_samples (Union[int,Tuple[int]]): Shape/size of samples to take.

        Returns:
            Tuple[np.npdarray]: Mean and standard deviation of samples, per row.
        """
        X = self._check_X(X)
        pred_dist = self.sample(X, n_samples)
        return pred_dist.mean(axis=0), pred_dist.std(axis=0)

    def predict_interval(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_samples: Union[int, Tuple[int]],
        hdi_prob: float = 0.95,
    ) -> np.ndarray:
        """
        Predict the highest-density interval for each data point, as generated from our
        n_samples model samples.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data/covariates.
            n_samples (Union[int,Tuple[int]]): Number of times to sample our function.
            hdi_prob (float, optional): The mass of probability to capture in our
            interval. Defaults to 0.95.

        Returns:
            np.ndarray: np.ndarray holding beginning, end points of interval.
        """
        X = self._check_X(X)
        preds = self.sample(X, n_samples).squeeze()
        return az.hdi(preds, hdi_prob=hdi_prob)
