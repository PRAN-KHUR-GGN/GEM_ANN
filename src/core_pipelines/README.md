# Core Pipelines Package

## Overview

The `core_pipelines` package is a Python package of (modular) pipelines for building modular pipelines for predictive and prescriptive modeling within OptimusAI studies.
Included here are both the definitions of nodes and the modular pipeline definitions themselves.

In addition, there are `kedro_utilities` helpful for common tasks, including for data validation with `great_expectations` custom `TagDict` kedro datasets, and custom kedro hook implementations.

Users are encouraged to leverage these components to build solutions they need to solve their client problems.

## Installation

### From the nexus artifact repository
1) Assuming .whl for pipelines + optimizer + optimus_core + pai are in `../packages`, 
`pip install core_pipelines --extra-index-url https://nexus-repo.mckinsey-solutions.com/nexus/repository/opm-pypi-releases/simple`


### From source (for core developers)
2) `python setup.py install` or in editable mode `pip install -e .`


### Optional Dependencies
This package contains modular pipelines which depend on pyspark, and a custom kedro hook implementation that requires PerformanceAI.
Trying to import and use these features without having installed either of these dependencies will throw ImportErrors.
Please use the optional install syntax of python to install the optional dependencies.
1) For example, the command to install the pyspark dependencies is: `pip install pipelines['pyspark'] --extra-index-url https://nexus-repo.mckinsey-solutions.com/nexus/repository/opm-pypi-releases/simple`


## Frequently Asked Questions (FAQ)
1) What do I do if the pipeline doesn't do what I want?
    - Extend it! Feel free to mix-and match the provided functionality with custom functionality. For example, you may 
    have a `project_train_model` pipeline that's constructed from some nodes in `core_pipelines` and other bespoke (potentially reporting?) 
    nodes that you write *just* for your project.
1) How can I re-use only a node or sub-set of nodes? 
    - Import that node from the library. In the past this import would have looked like `from optimus_pkg.pipelines.train_model.nodes import XXX`,
    now it would be `from core_pipelines.pipelines.train_moodel.nodes import XXX`.
1) Where do I get configuration for my pipeline? 
    - The start-up template will lay-down a sample config for your project type, but the `create_pipeline` function will
    also include a sample block of params to aid with config.
1) It sounds like I'm using a proprietary tool, how do I conduct a code hand-over with my client?
    - The `core_pipelines` package is a convenient way for building upon and handling code that enables versioning and updating.
    You will always have access to the latest master "source" code of the `core_pipelines` *package* on the optimus repo, and 
    will also have the .whl file for your project that you are free to hand-over with the other code of your project.
1) How do I use the pipelines and code present here?
    - The same way you've been using modular pipelines in the past, by importing! You are free to: use the modular pipelines as 
    is by importing the `create_pipeline` function from a given `pipeline.py` file within a modular pipeline directory, use only
    a subset of the nodes of this pipeline, import the nodes and use them within a new modular pipeline. These are only suggestions, 
    you are free to use the functionality presented here however best helps solve your client project. You will get the most mileage by *extending* and appending to this functionality, rather than directly modifying the provided source code in this python package.  This will enable you to more easily update and upgrade to more recent versions of Optimus.
     
