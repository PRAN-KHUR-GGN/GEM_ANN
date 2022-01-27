# Sensitivity Modular Pipeline

When conducting predictive-prescriptive projects (like Optimus), understanding of how the predictive model (objective function) takes on different values while varying control conditions is critical at multiple project steps.

Sensitivity data and charts help to improve predictive modeling by quickly surfacing any physical relationships which the model incorrectly captures. Moreover, such a visualization can help to increase operator and client buy-in by helping operators to understand the logic and relationships which the model has learned.

On the optimization side, sensitivity charts and sensitivity data can help when making choices on which recommendations should be accepted. For example, an operator might chose to set a control at a value different from a recommendation if the sensitivity chart is relatively flat in the region of the recommendation, which implies that the objective function would not change much by tweaking that single control.

## Overview

The main pipeline in the sensitivity modular pipeline creates a dataframe to enable later creation of sensitivity charts as described above. The produced data is at the recommendation level, and provides the estimated objective and control tag values, holding all other variables constant. CSTs can use this data in 3 ways:

1. To create static visuals themselves
2. To leverage the sensitivity charts within the UI.
3. To interactively query the objective function (predictive model) in the bundled model "simulator". The model simulator is a `streamlit` application which can be found in the same directory as this pipeline, and which creates an interactive sensitivity chart. The user can use this to view the objective function values while dynamically tweaking control and state values.

## Parameters

The sensitivity pipeline requires a small number of parameters to run. In particular, users need to specify the default resolution of the plot, as well as the columns to use as unique ids when specifying a block/shift and it's associated recommendations. If using neural network predictive models, users may need to also specify a dictionary of extra keyword arguments.

When leveraging the interactive model simulator, users will need to specify a mapping of dataset names. The model simulator loads a few generic data sources to memory when it runs, and depending on their use-case users may need to change the datasource names in order to point to the appropriate quantities in their respective data catalog.


```yaml
recommend_sensitivity:
    n_points: 50 # Resolution/number of objective values to plot when the tagdict doesn't specify a constraint set of values.
    unique_ids: # The unique columns which help to identify a set of recommendations.
        - run_id
        - timestamp
    objective_kwargs: {} # When performing counterfactuals with neural networks, these may need to be specified.
    percentile_range: [ -50, 50] # When this key is present, the calculated sensitivity will be given for relative changes of control variables.
    sensitivity_app_data_mapping: # Datasets to map when using the streamlit application.
        features: test_static_features # Name of the dataset holding features to load.
        model: train_trainset_model # Name of the dataset holding the objective to load.
        recs: test_recommendations # Name of the dataset holding the recommendations to load.
        sensitivity_data: test_sensitivity_plot_df # Dataframe of sensitivity data to load.
        timestamp_col: timestamp # Column id of the timestamp column
```

## Data Sets
The sensitivity pipeline must be run *after* the optimization/recommend pipeline. This is because of the dependency the sensitivity pipeline has on the current recommendations. These must be known in order to properly evaluate the counterfactual of what would happen if the control values are changed independently.

### Inputs
The sensitivity pipeline requires the following input:

- `td`: The tag dictionary.
- `input_model`: The predictive model which serves as objective function for optimization.
- `input_data`: The dataframe of features to be used for the optimization/counterfactuals. These data were not used for model building.
- `recommendations`: The set of recommendations that were found when running the `recommend` modular pipeline on `sensitivity.input_data`.


### Outputs
The pipeline creates the following outputs:
- `sensitivity_plot_df`: A dataframe that holds data for creating sensitivity charts for every recommendation-control combination present in input data. Running this pipeline for 1 recommendation that's provided for two controls will result in a sensitivity plot dataframe that can be used to create two curves, of the `objective` vs. `control_1` and of the `objective`
vs. `control_2`.

### Intermediate Outputs
The sensitivity pipeline has only one node, and as such has no intermediate outputs.

## Notebook example

You can find an example notebook for this pipeline within the examples folder of [core_pipelines](https://github.com/McK-Internal/optimus/tree/master/core_pipelines).
