# PyPlot_Reports pipeline

This is a modular pipeline to generate a PDF or Jupyter Notebook report build from MatPlotLib figures.

## Overview

During any data project many different charts and figures are created. Quite often these are lost, and are not easy to share with clients to generate discussion. This modular pipeline provides a way to output collections of these plots as either PDF documents, or as Jupyter Notebook outputs.

This modular pipeline can accept a list of any MatPlotLib charts (pyplot, seaborn, pandas, etc), and stitch them together into a clean, well formatted output. Two default styles are provided, light, and dark, but you can also edit this to create custom styles for your client or study.

## Parameters

You should speicify the parameters for the split pipeline in a `parameters.yml` file inside the `/conf` directory of your project.


### PDF

Generating a PDF report expects the following parameters:

```yaml
pdf:
    output_path: data/base/outputs     # Directory the report will be created
    report_name: eda                   # Name of the report - used to generate the filename
    timestamp: True                    # Flag indicating if you want to timestamp filenames
    title: "My Report"                 # Title of your report
    author: "My Name / CST Name"       # Author(s) of your report
    subject: "Latest outputs"          # Subject of your report
```

### Notebook

Generating a Notebook report expects the following parameters:

```yaml
notebook:
    output_file: data/base/output.ipybb     # Filename that report will be output to
    title: "My Report"                      # Title of your report
    author: "My Name / CST Name"            # Author(s) of your report
    subject: "Latest outputs"               # Subject of your report
```

Note that the notebook format does not support automatically adding timestamps to your filenames. This is because notebooks may be re-run at any point in time, reloading data from your catalog, which would invalidate the timestamp.

## Datasets
The pipeline expect a list of any number of MatPlotLib figures, which you should store in your catalog using the `pickle.PickleDataSet` format.

### Inputs
The pipeline inputs are generated dynamically based on the list of figures that you provide.

### Outputs
The pipeline does not return any output since the file is written locally based on params.

### Intermediate outputs
The pipeline does not have any intermediate outputs.

## Example usage

Let's consider that we have the following entries in our catalog:

```yaml
correlation_plot:
  type: pickle.PickleDataSet
  filepath: data/base/eda/correlation_plot.mpl

scatter_plot:
  type: pickle.PickleDataSet
  filepath: data/base/eda/scatter_plot.mpl
```

We can create our PDF and Jupyter Notebook reports using the following pipeline:

```python
from core_pipelines.pipelines import pyplot_reports

pipeline(
    pipe=pyplot_reports.create_pipeline(
        name_prefix="my_pipeline",
        plots=["correlation_plot", "scatter_plot"]
    ),
    parameters={
        "params:pyplot_reports.pdf": "params:pdf",
        "params:pyplot_reports.notebook": "params:notebook",
    }
),
```
