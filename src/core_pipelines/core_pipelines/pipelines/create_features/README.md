# Create features

The `create_features` pipeline creates temporal features from time-series data.

## Overview

Feature engineering, also known as feature creation, is the process of constructing new features from existing data before training a model. New features can be created by decomposing existing features, from external data sources, or by aggregating existing features. In this pipeline, we focus on feature creation for time series data. Time series data-points are aggregated into forwards and backwards looking features.<br/>
In order to be able to capture data at different time intervals and with gaps, this is done in the following steps:

1. Create a grid of timepoints with fixed frequency indicating all points at which features should be created
1. For each data set: add all grid points which are not already included in the data set, generate forward and backward looking features for all time points and finally reduce to only the timestamps on the grid
1. Merge all datasets : the datasets are merged column-wise on timestamp column

### Forwards vs backwards looking windows
The best aggregation method for time series data is highly dependent on the case at hand. Within Optimus, the feature creation and aggregation approach distinguishes between the model training and the optimization phases.
### Model training
During model training, at each point in time `t` we want to create:

* the best estimate of the state of the factory that we could have had at `t`
* the controls that were chosen from this point onwards
* the output that was achieved given those controls

Hence, we create backwards-looking state features and forwards-looking controls and targets.

### Optimization
When searching for optimal control parameters, at each point in time `t` we want to create:

* the best estimate of the state of the factory that we could have had at `t`
* the best estimate of our current control settings (note that theses are used only to present results; during optimization they are replaced by the parameters suggested by the solver)
* the output that was achieved given those controls (again, these are only used when estimating the outcomes of our optimization)

Hence, we create backwards-looking state features and controls with forwards-looking targets.

Below table summarizes the above points :


| &nbsp; 	| modelling 	| optimization 	|
|:-:	|:-:	|:-:	|
| state 	| backward-looking 	| backward-looking 	|
| control 	| forward-looking 	| backward-looking 	|
| targets 	| forward-looking 	| forward-looking 	|


## Parameters

Following are the parameters used in this pipeline


```yaml
create_features:
    pipeline_timezone: "UTC"      # The timezone used to set timestamps variables
    n_jobs: 6                     # Indicates the parallelization count
    grid:
        frequency: "1H"           # Indicates feature grid frequency (pandas frequency code), e.g. `1H` leads to an hourly grid.
        offset_start: "2H"        # Indicates the time between first timestamp found in the data and first grid point
        offset_end: "2H"          # Indicates the time between last timestamp found in the data and last grid point

```
Note : Frequnecy of time-stamp grid can have larger values as well. For e.g. `1D` denoting 1 day frequent grids. For more info refer to pandas api : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html

### Tag dictionary related configuration
A key thing in context to this pipeline is some required attributes in the tag dictionary. It must reflect following tag attributes like :

* `agg_window_length` : length of window over which to aggregate during static feature creation - use pandas frequency codes such as `1H` or `30 min`
* `agg_method` : aggregation method, e.g. `mean`, `sum` or `mode`.

Note that some aggregations are much more susceptible to noise in the timestamps than others (e.g. `sum` vs `median`).
Since grid and aggregation are decoupled, it is perfectly fine to create features e.g. once an hour but only look back for 10 min for some of them. In order to prevent the transmissions of redundant information, users should take care to avoid overlapping observations.

## Data Sets
The `create_features` modular pipeline has 2 inputs which a user must supply. The first is a list of the data sources on which features should be created. The second is a method of feature generation, currently two methods are supported: `create_train_features` and `create_opt_features`. These are described in further detail later on.

### Inputs
The pipeline requires the following input datasets:

* `fit_data`: dataset for which features need to be created
* `td`: Tag dictionary object

### Outputs
The pipeline creates the following output datasets:

* `static_features`: merged dataset with static features ( when used for training section )
* `output` : merged dataset with static features and uuid ( when used for optimization section )

### Intermediate Outputs
The pipeline creates the following intermediate output datasets:

* `time_grid` : time grid generated for the given dataset
* `static_features`: static features dataset with no uuid

## Notebook example

You can find an example notebook for this pipeline within the examples folder of [core_pipelines](https://github.com/McK-Internal/optimus/tree/master/core_pipelines).
