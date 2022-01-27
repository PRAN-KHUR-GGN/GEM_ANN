# Split Pipeline

Pandas pipeline to split data based on a datetime. This split can be based on a specific datetime, or by fraction.

## Overview

When working with temporal data, a common task is the split the data into a train and test set based on time. The train set is used to train some data science model, and then the test set is used to evaluate the model performance. It is important that all data in the test set comes temporally after data in the train set, so that the data scientist can make claims about how well their model will generalise into the future.

__Datetime__
For data which spans a fixed, known time-window, the data scientist may want to provide an explicit split point. We refer to this as a “datetime” split.

__Fractional__
For data which spans an unknown time-window, it is more natural to think about splitting the data into fractional parts. For example, using the first 70% of time as the train set, and the remaining 30% of time as the test set. We refer to this as a “fraction” split.

__Interval__
For temporal data, we may want to have granular control over the specific, potentially disjoint date-ranges comprising our train and test sets. For example, we might have a range of days over which we know there were certain plant conditions. We refer to this as a "periods" split.

## Parameters

You should specify the parameters for the split pipeline in a `parameters.yml` file inside the `/conf` directory of your project.


### Split By Fraction

When splitting data by fraction, ordered by date, the following parameters are required.

```yaml
split:
    type: 'frac'              # Indicates the split should be based on a percentage of the data
    datetime_col: 'timestamp' # The datetime column name in the dataset
    train_split_fract: 0.9    # This indicates the percentage split between train and test dataset
```

### Split By Datetime

When splitting data by a specific datetime, the following parameters are required.

```yaml
split:
    type: 'date'                                     # Indicates the split should be based on a specific date
    datetime_col: 'timestamp'                        # The datetime column name in the dataset
    datetime_val: !!timestamp '2001-04-26 03:59:59'  # This datetime is used for splitting the data
```
### Split By Periods (intervals)

When splitting data by a specified datetime intervals, the following parameters are required.

```yaml
split:
    type: 'periods'                                  # Indicates the split should be based on intervals
    datetime_col: 'timestamp'                        # The datetime column name in the dataset
    train_splits: # datetime intervals defining the training set
      - [!!timestamp '2001-04-26 00:00:00', !!timestamp "2001-05-01 00:00:00"]
      - [!!timestamp '2001-06-15 00:00:00', !!timestamp "2001-06-20 00:00:00"] 
    test_splits: # Intervals defining the test set
      - [!!timestamp '2001-08-26 00:00:00', !!timestamp "2001-09-01 00:00:00"]
```



## Data Sets
The split pipeline expects a single dataset as an input, and returns a train and test split of this dataset as an output.

### Inputs
The split pipeline requires the following input:
- `split.data`: The input dataset that should be split into train and test.

### Outputs
The split pipeline creates the following outputs:
- `train`: The train part of the dataset
- `test`: The test part of the dataset

### Intermediate Outputs
The split pipeline does not have any intermediate outputs.

## Notebook example

You can find an example notebook for this pipeline within the examples folder of [core_pipelines](https://github.com/McK-Internal/optimus/tree/master/core_pipelines).
