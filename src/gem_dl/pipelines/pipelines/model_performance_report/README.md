# Model Performance Report

The model_performance_report pipeline is a modular pipeline for generating reports summarizing the 
performance of a trained predictive model. It requires artifacts (detailed below) that will (often) 
be generated from running with the train_model modular pipeline.

## Overview

This pipeline generates figures, as well as a client-ready pdf report summarizing a single predictive model's
performance.

## Parameters
The following parameters are used in the train_model pipeline.
```yaml
model_performance_report:
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

## Data Sets
The train model pipeline expects an input dataset and a tag dictionary. It stores the model object, cross-validation results, feature importance, test-set predictions and evaluation metrics.

### Inputs
The pipeline requires the following input:
- `model`: (Trained) model object
- `feature_importance`: Feature importance for the developed model
- `train_set_metrics`: Performance evaluation metrics from the training data
- `train_set_predictions`: Predictions from the training data.
- `test_set_metrics`: Performance evaluation metrics from the test data
- `test_set_predictions`: Predictions from the test data.
- `static_features`: Features used to build model

### Outputs
The pipeline creates the following outputs:
- `report_figures`: A dictionary of matplotlib.pyplot figures that is the content of the generated report.