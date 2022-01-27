# Stitch Time-Series

## Overview
Raw data often comes from multiple sources or files/tables. Before working on it you would want to join these datasets together to form one _master table_ to process this data down the road.

The stitch_timeseries pipeline takes multiple datasets and joins them on a common _time grid_. A time grid is a timestamp index with a consistent frequency.


## Dependencies
This module uses the `resample` modular pipeline to apply resampling logic when merging datasets.

## Pitfalls when joining timestamped data
Joining raw timeseries data onto a (different) time grid can distort your data quite substantially. Common issues and approaches to mitigate are listed below.

Given are two timestamp-indexed datasets with different base frequencies:

A 1-minute indexed __df1__:

| timestamp           |   col_1min    |
|:--------------------|--------------:|
| 2021-01-01 00:00:00 |            10 |
| 2021-01-01 00:01:00 |            20 |
| 2021-01-01 00:02:00 |            30 |
| 2021-01-01 00:03:00 |            40 |
| 2021-01-01 00:04:00 |            50 |
| 2021-01-01 00:05:00 |            60 |
<br />

A 5-minute indexed __df2__:

| timestamp           |   col_5min    |
|:--------------------|--------------:|
| 2021-01-01 00:00:00 |            50 |
| 2021-01-01 00:05:00 |            50 |
| 2021-01-01 00:10:00 |            50 |
<br />

### Joining on higher frequency timestamp index
Joining __df2__ to __df1__ will create NaN in the resulting frame:

| timestamp           |   col_1min    |col_5min|
|:--------------------|--------------:|---------:|
| 2021-01-01 00:00:00 |            10 |50|
| 2021-01-01 00:01:00 |            20 |NaN|
| 2021-01-01 00:02:00 |            30 |NaN|
| 2021-01-01 00:03:00 |            40 |NaN|
| 2021-01-01 00:04:00 |            50 |50|
| 2021-01-01 00:05:00 |            60 |NaN|
<br />

This case needs to be addressed with appropriate interpolation or filling to treat the resulting holes. There are other pipelines designed to interpolate missing values that should be applied after joining a on a higher frequency timestamp index.

### Joining on lower frequency timestamp index
Joining __df1__ to __df2__ will "hide" the none matching samples. This is a problem if the data represents a continuous quantity or an average of the time window represented by the timestamp ("1 minute average temperature"):

| timestamp           |   col_5min    |col_1min|
|:--------------------|--------------:|---------:|
| 2021-01-01 00:00:00 |            50 |10|
| 2021-01-01 00:05:00 |            50 |60|
| 2021-01-01 00:10:00 |            50 |NaN|
<br />

The value at timestamp `2021-01-01 00:00:00` is now "stretched" over the whole time window from `00:00:00` to `00:05:00`, all other values `00:01:00`, `00:02:00` and so forth are dropped. A better representation would have been given by an aggregation of `col_1_min` between `00:00:00` and `00:05:00`, e.g. the average, of all the samples :

| timestamp           |   col_5min    |col_1min|
|:--------------------|--------------:|---------:|
| 2021-01-01 00:00:00 |            50 |30|
| 2021-01-01 00:05:00 |            50 |NaN|
<br />

Joining a lower frequency timestamp to a higher frequency timestamp requires appropriate resampling and aggregation.

The `stitch_timeseries` pipeline offers a parameter that allows you to `resample` a dataframe before joining it to the time grid.

### Joining non-matching timestamp indices
Let us assume a third dataset with an uneven timestamp:
__df3__

| timestamp           |   col_5min_uneven    |
|:--------------------|--------------:|
| 2021-01-01 00:01:32 |            100 |
| 2021-01-01 00:04:59 |            200 |
| 2021-01-01 00:09:00 |            300 |
<br />

Joining that to __df2__ will result in NaNs:

| timestamp           |   col_5min    |col_5min_uneven|
|:--------------------|--------------:|---------:|
| 2021-01-01 00:00:00 |            50 |NaN|
| 2021-01-01 00:05:00 |            50 |NaN|
| 2021-01-01 00:10:00 |            50 |NaN|
<br />

This can be partially mitigated by rounding the timestamps to full units (e.g. minutes). The `stitch_timeseries` pipeline also offers a parameter to do a `merge_asof` instead of a left join, which will map the uneven indexed data to the _closest_ timestamp in the resulting frame:

| timestamp           |   col_5min    |col_5min_uneven|
|:--------------------|--------------:|---------:|
| 2021-01-01 00:00:00 |            50 |100|
| 2021-01-01 00:05:00 |            50 |200|
| 2021-01-01 00:10:00 |            50 |300|
<br />

### Joining absolute quantities on different timestamp index
If a column represents an absolute quantity for the given timespan (e.g. metric tons) instead of a continuous/relative quantity (e.g. temperature, tons per hour), joining on a different timeseries index distorts the meaning of the sample:

| timestamp           |   tons produced  |
|:--------------------|--------------:|
| 2021-01-01 00:00:00 |            50 |
| 2021-01-01 00:05:00 |            60 |
| 2021-01-01 00:10:00 |            100 |
<br />

In this example case the column describes the absolute tons of production for the time window, e.g. "50 tons produced in the window between `00:00:00` and `00:05:00`.

Joining this to a higher frequency timestamp (e.g. 1 min) and forward filling will distort the meaning of the column. Joining on a lower frequency timestamp (e.g. 30 min) is possible by resampling and using `sum` as an aggregation method.

Generally, absolute values can be avoided by calculating the relative equivalent, i.e. "tons produced per minute".

## Parameters
The following parameters are used in the stitch_timeseries pipeline.
```yaml
stitch_timeseries:
    pipeline_timezone: "UTC"           # The timezone used to set timestamps variables
    master_timestamp_col: "timestamp"  # The name of the time grid timestamp column
    grid:
        frequency: "15T"       # Indicates frequency of target joined time grid
        offset_start: "15T"   # Offset to add to  first timestamp in grid
        offset_end: "15T"   # Offset to add to last timestamp in grid
    sources: # A list of dictionaries specifying how to join
        - name: input_1 # kedro datasource name
        merge_strategy: "match"                    # Strategy to apply to <input_name> dataset when joining to the grid. Can be {"match", "merge_asof", "resample}. Match does a left join on grid timestamp. Merge asof matches to the closest grid timestamp. Resample applies the aggregation logic defined in agg_method in the data dictionary to resample each column to time grid frequency before left joining.
        datetime_col: "sample_status_time" #name of time column in the dataset
```

## Data Sets

### Inputs
All data source names to be joined need to be provided to the pipeline when calling the `create_pipeline` method:
```python
stitch_timeseries.create_pipeline(input_data=["input_1", "input_2", "input_n"])
```
This will set up the pipeline to expect `input_1`, `input_2`, `input_n` and so forth as input datasets:

* `input_1` - the first dataset to be joined
* `input_2` - the second dataset to be joined
* `input_n` - the n-th dataset to be joined

### Outputs
The pipeline returns a single joined dataset, the _master table_ as an output (`output`).


## Pipeline nodes
The `stitch_timeseries` pipeline has two nodes:
- `create_time_grid`: Creates a regular time grid
- `merge_to_grid`: Merges all data sources to the time grid

## Pipeline cheatsheet

To execute the pipeline in a separate Kedro project, copy the parameters into `conf/base/parameters.yml`:
> Note: The dataset specific inputs have to be renamed/extended to your input data names as defined above (`input_1` to `input_n`)

> Note: The pipeline expects a specific yaml element of the same name as each of the inputs defined when calling `create_pipeline`, here `input_1` and `input_2`
```yaml
stitch_timeseries:
    pipeline_timezone: "UTC" # see globals
    grid:
        frequency: "15T"
        offset_start: "15T"
        offset_end: "15T"
    sources:
        - name: input_1
          merge_strategy: "match" # 'match', 'merge_asof' or 'resample'
        - name: input_2
          merge_strategy: "match" # 'match', 'merge_asof' or 'resample'
```

Input parameters can also be shared across elements by applying yaml syntax. Here, the parameters defined in `common` are extended to the two input datasets `input_1` and `input_2`:
```yaml
stitch_timeseries:
    pipeline_timezone: "UTC" # see globals
    grid:
        frequency: "15T"
        offset_start: "15T"
        offset_end: "15T"
    common: &stitch_common
        merge_strategy: "match" # 'match', 'merge_asof' or 'resample'
    sources:
        - name: input_1
          <<: *stitch_common
        - name: input_2
          <<: *stitch_common

```

Copy this pipeline into your `pipeline.py` (don't forget to add additional datasets to join, if any):

```python
from kedro.pipeline import pipeline
from core_pipelines.pipelines import stitch_timeseries

my_stitch_pipeline = pipeline(
    pipe=stitch_timeseries.create_pipeline(
        input_data=[
            "input_1",
            "input_2",
            ],
        ),
        inputs={
            "input_1": "input_1_cleaned",
            "input_2": "input_2_cleaned",
        },
        outputs={
            "output": "master_table"
        },
        parameters={"params":"params:stitch_timeseries"},
        namespace="master_table",
    )
```
