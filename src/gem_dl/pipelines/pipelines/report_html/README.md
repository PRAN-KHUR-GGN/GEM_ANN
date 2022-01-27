# Pipeline report_html (In Development)

## Overview

This pipeline allows users to create an html report for figures provided.

The pipeline uses the `template.md` file provided in this module to render the figure names,
images and descriptions. It is based on a Jinja 2.0 template and users can modify
it if needed.

## Pipeline inputs

The pipeline expects two inputs:

1. A list of dictionary of figures.

Example:

```python
figures = [
    {"name": "scatter_plot",
     "path": "use_case_pipelines/predictive_modeling/data/08_reporting/download.png"},
    {"name": "bar_chart",
     "path": "use_case_pipelines/predictive_modeling/data/08_reporting/bar_chart1"
             ".png"},
    {"name": "correlation_plot",
     "path": "use_case_pipelines/predictive_modeling/data/08_reporting/correlation_plot"
             ".png"}
]
```

2. Parameters input

All the paths are relative to your local kedro project. If the template is at a
different location, or you would like to save the report to a different location,
this can be done by adding a relative path from the local project.

Example:

```yaml
reporting:
  meta_data:
    "author": "OAI"
    "time": "04-12-2022"
    "title": "Example Report"
  report_path: "data/08_reporting/report_output.html"
  report_template_path: ""
  template_markdown: "template.md"
  fig_desc:
    correlation_plot: "This is a correlation plot to understand how are the features correlated"
```

## Pipeline outputs

An html file is generated at the specified path in `report_path` parameter..

Note if the images are not being rendered properly in the output file, please check
that the path of the images is correct.
