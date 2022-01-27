# Reporting HTML

## Overview

Jupyter notebooks provide a great platform for enhanced reporting. The `reporting_html` pipeline allows you to run jupyter notebooks from within a Kedro pipeline and output the notebook to a .html report.

You can define fully customized notebook templates and feed them to this pipeline.

## Assumptions

The notebooks executed with this pipeline have to be self-contained. You can not feed any data or parameter from the pipeline directly to the notebook template.

To be useful in a pipeline context, the notebook template has to:
- Load the kedro context
- Explicitly load a certain dataset available in the data catalog at runtime
- Load parameters at runtime from the `parameters.yml`

## Parameters

A small number of parameters are required for using the reporting_html pipeline. An example params block is:

```yaml
# for pipelines creating multiple reports, it's useful to define a common block
html_report_options: &html_report_common
    kernel: "python3"
    timeout: 600 # number of seconds before execution times out.
    remove_code: True # remove code from html result.
    timestamp: True # When True, adds a timestamp to filename

interactive_eda_raw_report:
    <<: *html_report_common
    # Path in your project to the template file
    template_path: src/<project_name>/ipynb_templates/interactive_variable_report_template.ipynb
    # where to store the output
    output_dir: data/08_reporting/
    # output file name
    report_name: "variable_exploration_report_raw"

```

## Data Sets

No explicit dataset inputs required

### Inputs
The `create_pipeline` method of this pipeline has to be called with a list of datasets that you want to wait on being generated before running the reporting html pipeline. This avoids that the report is trying to access a dataset that has not yet been generated or updated.


```python
reporting_html.create_pipeline(wait_on=["<dataset_name>"])
```
This example indicates that the pipeline needs to wait on the dataset `"<dataset_name>"` to be created before executing. 
All the datasets in `wait_on` have to be provided as inputs to the pipeline.

An example instantiation of this pipeline in a project might look like the following (building on the params above):
```python
from core_pipelines.pipelines import reporting_html
Pipeline(
    nodes=[
        pipeline(
            pipe=reporting_html.create_pipeline(wait_on=["joined"]),
            inputs={"joined": "joined"},
            namespace="interactive_eda_raw",
            parameters={
                "params:reporting_html": "params:interactive_eda_raw_report"
            },
        )
    ]
),

```

The pipeline requires the following input:
- `<dataset_name>`: The dataset(s) you specified to wait on

It is useful to store the report templates inside of a common folder in the project `src`, e.g.

```
/src/<your_pipeline_module_name>/ipynb_templates
```

### Outputs
The pipeline does not create an output dataset. It stores the generated report in
```
<reporting_dir>/<report_name>.html
```
For example:
```
/data/08_reporting/variable_exploration_report.html
```
