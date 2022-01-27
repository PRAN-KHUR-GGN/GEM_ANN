# Recommend

This pipeline is responsible for counterfactual generation, and returns results summarizing proposed control values and the estimated uplift.

## Overview
The `recommend` pipeline is critical for **parallelizing** counterfactual generation. Specifically, it assumes that the input dataset contains the rows corresponding to the unit of analysis (whether shift, block or something else) on which we would like to calculate potential uplift. The optimization for a given row is *vectorized* via the solver implementations within the `optimizer` package. This allows us to instead parallelize over each unit of analysis.

The `recommend` pipeline's single responsibility is to parallelize the application of an ask/tell loop over rows of an input dataset. By default, we leverage the OptimusAI `optimizer` package for this, but solver choice is a user-changed parameter, as discussed below. The `recommend` pipeline's single node contains the `bulk_optimize` function, which handles the creation of the solver, objective function, necessary constraints and repairs, and any stoppers. When it finishes this node returns a dictionary containing dataframes summarizing the recommended changes for each unit of analysis. These are later post-processed for viewing in the UI and/or debugging purposes.

### Frequently Asked Questions
1) What should I do if the recommend pipeline is taking a long time to complete?

- It's best to take a look at optimizing calls to your objective function itself. Specifically, the objective function should be as __vectorized__ as possible in the "batch" dimension. That is, evaluations should not incur computational complexity strongly depending on the number of potential solutions you're predicting or scoring on.

2) I've been running my pipeline overnight, and I see it has many counterfactuals still to complete, but would like to see pipeline results for however far it's gotten.

- The `bulk_optimize` and `bulk_optimize_context` functions used in the `recommend` pipeline can handle `KeyboardInterrupt` errors. When entered during a running optimization, the execution will terminate and the pipeline will continue with whatever counterfactuals have already been processed.

## Parameters

The following parameters are used in the `recommend` pipeline.

```yaml
recommend:
    solver: # Parameters for the solver class and any/all keyword arguments
        class: optimizer.solvers.DifferentialEvolutionSolver
        kwargs:
            sense: "maximize"
            seed: 0
            maxiter: 100
            mutation: [0.5, 1.0]
            recombination: 0.7
            strategy: "best1bin"
    stopper: # Parameters for any stoppers to help terminate optimization early
        class: optimizer.stoppers.NoImprovementStopper
        kwargs:
            patience: 10
            sense: "maximize"
            min_delta: 0.1
    n_jobs: 4 # How many workers to parallize over. Often set to ~ number of available cpus.
    objective_kwargs: {} # keyword args of your objective. Will likely only be used with RNN/sequence models
```

## Data Sets
The recommend pipeline expects the follow datasets (in addition to `params` mentioned above).

### Inputs
The pipeline requires the following input:
- `td`: Tag Dictionary
- `input_data`: Input dataframe holding our test, or _counterfactual_ set.
- `input_model`: The objective function, which can be any python object with a predict method. For some users, this will be a function which combines several other predictive models which have been previously trained.

It may be required to customize the behavior of your optimization problem, through specification of repairs, constraints, 
and bounds. We supply three example nodes that calculate default settings, but it may be necessary to over-ride this functionality.

In each case, the `bulk_optimize*` functions expect three datasets, called respectively:
- `penalty_dict`: Values are a list of penalty objects.
- `repairs_dict`: Values are a list of repair objects.
- `bounds_dict`: Values are a list of tuples representing solver domain bounds.
- `features`: A list of input_features to use for optimization.

The artifacts which are python dictionaries have keys being the index of the optimization input dataset. The values of each will
be python lists. By default these are created by iterating over the rows of the optimization dataset in order to generate
these objects, which had previously been done inside the `bulk_optimize` functions.

Any user-supplied utility can be used to generate new versions of these datasets if more functionality or more complex penalties
are required.

### Outputs
The pipeline creates the following outputs:
- `recommendations`: Dataframe holding a summary, for each row of input, of the `run_id`, `timestamp`, dictionary of state/non-optimizable variables, dictionary of recommended changes to optimizable variables/controls, dictionary of current, predicted current and optimized outcome variables, and a summary of penalties and slacks incurred.
- `recommended_controls`: Dataframe of the recommended values for each control
- `projected_optimization`: Dataframe the baseline and optimized values for the outcome variable `tag_id`.


## Notebook example

You can find an example notebook for this pipeline within the examples folder of [core_pipelines](https://github.com/McK-Internal/optimus/tree/master/core_pipelines).
