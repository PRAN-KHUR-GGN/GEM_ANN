# Train Model

The train model pipeline builds a model on the training data, cross validates it and uses the best model to predict on the test data.

## Overview

After data cleaning, data manipulation and feature generation, it is time to build a model. A good model is robust to overfitting, and is generalized enough to predict unseen data with good acuracy. Within the `train_model` pipeline, a model is built by first tuning it's hyper-parameters. By default, this is done via GridSearch k-fold cross-validation. The model with the best parameters is chosen to predict on the test dataset. The cross validation results, performance metrics and the model are stored as outputs.

## Dependencies
This module uses the `split` modular pipeline to split its input into train and test.

## Parameters
The following parameters are used in the train_model pipeline.
```yaml
train_model:
    split:
        type: frac                                         # Indicates the split should be based on a percentage of the data
        train_split_fract: 0.7                             # This indicates the percentage split between train and test dataset
        datetime_val: !!timestamp '2020-04-26 03:59:59'    # This datetime is used for splitting the data
        datetime_col: "timestamp"                          # The datetime column name in the dataset

    transformers:
        # Specify list of dicts, each holding specification and name of a
        # transformer, feature selection or not
        - class: sklearn.feature_selection.SelectKBest
          kwargs:
              score_func: sklearn.feature_selection.mutual_info_regression
              k: 7
          name: mutual_information_selector
          selector: True

    estimator:
        # This example configuration builds a small MLPRegressor from scikit learn.
        # For more advanced deep learning model builds, please use the dedicated
        # train_dl_model modular pipeline
        class: sklearn.neural_network.MLPRegressor
        kwargs:
            random_state: 42
            activation: relu
            hidden_layer_sizes: 15

    cv:
        class: sklearn.model_selection.KFold               # Name of the cross-validation method
        kwargs:
            n_splits: 5                                    # Number of folds for cross-validation
            random_state: null                             # Random seed. Relevant only when shuffle is True.

    tuner:
        class: sklearn.model_selection.GridSearchCV        # Name of the hyper-parameter tuning method
        kwargs:
            n_jobs: -1                                     # Number of parallel threads used
            refit: mae                                    # The scorer used to find the best parameters for refitting
            verbose: True                                  # Print information
            param_grid:
                estimator__hidden_layer_sizes: [ [ 15 ], [ 15,3 ] ]
                estimator__activation: [ 'identity', 'relu' ]            
            scoring:
                mae: neg_mean_absolute_error               # Scoring parameter name for mean absolute error
                rmse: neg_root_mean_squared_error          # Scoring parameter name for root mean squared error
                r2: r2                                     # Scoring parameter for R-square
    report:
        output_path: poc/train/08_reporting                # Location to save report
        report_name: performance_report                    # Name of performance report
        timestamp: True
        title: "Model Performance Report"                  # Metadata to place on cover page
        author: "OptimusAI"
        subject: "OptimusAI Performance Report"                
```

#### shap
`shap` is not installed by default, to leverage the shap reporting nodes available in `core_pipelines.pipelines.utils.shap_plots`, you can install shap either via `pip install shap` or via `pip install core_pipelines[shap]`. You can then add the `generate_shap_figures` to your pipeline - it will generate a dictionary of shap plots for your model. Additional functions for generating individual level shap plots are also available in the `shap_reports` module.

You need to pass extra params to this function, to do so, please add a block like the below to your params.

```yaml
shap:
    explainer: shap.DeepExplainer
    kwargs: {}
```
### Advanced Tuning Methods
Users can leverage advanced hyper-parameter tuning methods available through the [`tune_sklearn`](https://docs.ray.io/en/master/tune/api_docs/sklearn.html) library, which provides a 
scikit-learn interface to algorithms available in the `ray[tune]` package. To leverage these, the parameters for the tuner
block above will look slightly different. Users would need to install the extra `tune_sklearn` package.

An example block, which would tune the `max_depth`, `learning_rate`, and `reg_lambda` params of a XGBRegressor model is:

```yaml
tuner:
    class: tune_sklearn.TuneSearchCV
    kwargs:
        n_jobs: -1
        refit: mae
        verbose: True
        n_trials: 2
        max_iters: 10
        search_optimization: 'bohb'
        param_distributions:
            estimator__max_depth: [2,3,5,8]
            estimator__learning_rate: (0.05, 0.2, 'uniform')
            estimator__reg_lambda: [0.1, 1, 10]
        scoring:
            mae: neg_mean_absolute_error               # Scoring parameter name for mean absolute error
            rmse: neg_root_mean_squared_error          # Scoring parameter name for root mean squared error
            r2: r2                                     # Scoring parameter for R-square           
```

## Data Sets
The train model pipeline expects an input dataset and a tag dictionary. It stores the model object, cross-validation results, feature importance, test-set predictions and evaluation metrics.

### Inputs
The pipeline requires the following input:
- `input`: Input dataset to be modeled
- `td`: Tag dictionary

### Outputs
The pipeline creates the following outputs:
- `model`: Model object
- `train_set_model`: Model object on the training dataset.
- `train_set_cv_results`: Cross-validation results from hyper- tuning.
- `train_set_feature_importance`: Feature importance for the developed model
- `test_set_metrics`: Performance evaluation metrics from the test data
- `test_set_predictions`: Predictions from the test data.

### Intermediate Outputs
The pipeline creates the following intermediate outputs:
- `train_set`: Training set that is used to build model
- `test_set`: Test set that is used for prediction
- `estimator`: Model estimator with all parameters
- `estimator_pipeline`: Sklearn Pipeline with selected columns and model estimator

## Notebook example

You can find an example notebook for this pipeline within the examples folder of [core_pipelines](https://github.com/McK-Internal/optimus/tree/master/core_pipelines).
