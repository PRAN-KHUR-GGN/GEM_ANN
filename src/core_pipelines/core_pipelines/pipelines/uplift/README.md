# Uplift

The uplift pipeline helps in creating bulk result tables which can be used to create uplift reports after recommendations are generated via optimize pipeline.

## Overview

After optimization is done, and recommendations are generated, one needs to create a bulk-optimization output table. This report is a flat file and has current and suggested values for all variables. The constraints information is also included which is helpful to debug and understand the optimized solution better. Additionally, separate tables for state variables, control variables, outcome variables and constraints are stored as pickle files, which can be further processed as needed by the user. These bulk tables are used to generate an uplift report.


## Parameters
A typical parameters block is the following. It is used to parameterize the reporting function (final node).
```yaml

uplift:
    report:
        output_path: test/08_reporting/ #location to save report
        report_name: uplift_report
        timestamp: True
        title: "Uplift Report"
        author: "OptimusAI"
        subject: "OptimusAI Uplift Report"

```

## Data Sets
The uplift pipeline requires a tag dictionary, an input file and recommmendations file.

### Inputs
The pipeline requires the following input:
- `td` : Tag dictionary
- `params`: Parameters for uplift reporting
- `input_data` : Data used for optimization pipeline
- `recommendations` : Data output from optimization pipeline

### Outputs
The pipeline creates one output:
- `bulk_report` : bulk report - with current and suggested values for all variables.

### Intermediate Outputs
The pipeline creates the following intermediate outputs. These are stored as pickled datafiles.
- `bulk_state` : bulk table for state variables,
- `bulk_ctrl` : bulk table for contorl variables,
- `bulk_output` : bulk table for output variables,
- `bulk_constraints` : bulk table for constraints information i.e penalty and slack

## Notebook example

You can find an example notebook for this pipeline within the examples folder of [core_pipelines](https://github.com/McK-Internal/optimus/tree/master/core_pipelines).
