Welcome to this README.md file.

To run this notebook, use the section at the bottom of this notebook to set up a dummy catalog and parameters, as used in this REAME.

This cell is hidden in built docs

# Time Shifting

## Overview

Depending on the complexity of the system a CST is looking to model, you may need to account for the amount of time material takes to move through a system. In situations like this you'll need to time shift your data to account for this lag/lead for key sensors.

For example, if you are looking to optimize the recovery of oil at a refinery and are trying to provide a recommendation every hour; you may need to shift the total amount of input material forward by 1 hour assuming it takes 1 hour to move through the system before being refined into the final product.

The time shifting module helps with this situation by taking in parameters on a column by column basis and applying the time shift for those given columns while leaving columns that do not require a shift alone.

## Module Assumptions
The time shifting module makes a few key assumptions about your data:

- Your dataset has a datetime index called `timestamp` - this can be over-ridden through parameters

- Your data has already been aggregated from its raw format up. E.g. 10 records between minute 1 and minute 2 get aggregated up in to one record which summarizes all data in between that time.

- Your input table frequency is at a lower granularity than the time shift amount. E.g. My table is at the 10 minute level and the amount you wish to time shift a column is 20 minutes

- Your time shifting amount must be a multiple of your table frequency. For example, if your table frequency is 10 minutes, but you want to time shift by 15 minutes, **you cannot do this since 12:00am time shifted 15 minutes would be equal to 12:15am. Since your table frequency is 10 mins, 12:15am would not exist in your table.** In this case you would have to shift by 10 mins/20 mins/30 mins/40 mins/etc.

- You are not time shifting data less than 1 minute. Currently the time shifting module does not support sub minute shifts.


## Key Note
The time shifting module does all of it's arithmetic in minutes. So if you wish to time shift a column forward 1 hour, your parameter for the corresponding column would be `{'sensor_1': 60}`.

Alternatively, if you wish to shift 1 hour backward, the parameter value would be `{'sensor_1': -60}`

## Determining Appropriate Time Shifts
There are three options for determining the appropriate time shifts on a sensor by sensor basis.

- Use autocorrelations in order to determine the best time shift to use on a given sensor

- Meet with SME's who have in depth knowledge of the system you are modeling and ask them how long to shift by.

- The best method is a combination of the two. A data scientist on the team should do an initial analsis to find the appropriate shift time, but then take these numbers for validation to the SME team.

## Additional Considerations
Since you are time shifting your data, the beginning and end of your data frame will introduce Null/NaN values. Consider imputing/interpolating your data at these edge cases.

Note: This will only affect data in the very beginning or the very end of your data. For an example of this, refer to the output dataframe under the "Time Shifted Output" header.

## Parameters

Pipeline parameters


Below is an example of what your parameters file would look like for the time shifting module.


```yaml
time_shifting:
    timestamp_col: "${master_timestamp_col}"
    pipeline_timezone: "${pipeline_timezone}"
    data_frame_frequency: 15
    cols:
        col_1: 60
        col_2: -60
        dummy_col: -15
```


Sensor 1 corresponds to a 1 hour shift forward.

Sensor 2 corresponds to a 1 hour shift backward.

Sensor 3 corresponds to a 15 minute shift backward.

## Data Sets
The time shifting pipeline expects two inputs - the dataset which has columns that you wish to shift and the input parameters

### Inputs
The impute pipeline requires the following input:

- `parameters`: Tag dictionary object

- `time_shift_demo_data`: the data whose values need to be imputed

### Outputs
The pipeline creates the following output:

- `time_shifted_data`: the dataset with time shifted data values


## Notebook example

You can find an example notebook for this pipeline within the examples folder of [core_pipelines](https://github.com/McK-Internal/optimus/tree/master/core_pipelines).
