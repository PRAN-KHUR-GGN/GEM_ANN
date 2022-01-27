from kedro.pipeline import Pipeline, node

from .nodes import preprocess_energy


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_energy,
                inputs="energy",
                outputs="preprocessed_energy",
                name="preprocess_energy_node",
            ),

        ]
    )
