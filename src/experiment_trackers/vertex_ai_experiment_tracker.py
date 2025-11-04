"""An experiment tracker for Google Cloud Vertex AI."""

import hypertune
from google.cloud import aiplatform

from src.environment import ENVIRONMENT
from src.experiment_trackers.experiment_tracker import (
    ExperimentTracker,
    ExperimentTrackerBuilder,
)


class VertexAIExperimentTracker(ExperimentTracker):
    """
    Logs parameters and metrics to Vertex AI Experiments.
    Reports HPO metrics using hypertune.
    """

    def log_params(self, params: dict):
        aiplatform.log_params(params)

    def log_metrics(self, metrics: dict):
        aiplatform.log_metrics(metrics)

    def report_hpo_metric(self, metric_name: str, metric_value: float, step: int):
        """
        Reports the HPO metric using the cloudml-hypertune library.
        """
        # pylint: disable=no-member
        hpt = hypertune.HyperTune()
        # pylint: enable=no-member

        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=metric_name,
            metric_value=metric_value,
            global_step=step,
        )


class VertexAIExperimentTrackerBuilder(ExperimentTrackerBuilder):
    """
    Builds and initializes the VertexAIExperimentTracker.
    """

    RUN_NAME = ENVIRONMENT.get(ENVIRONMENT.VARIABLES.RUN_NAME)

    @property
    def argparser(self):
        argparser = super().argparser
        argparser.add_argument(
            "--vertex-project-id",
            type=str,
            required=True,
            help="Google Cloud Project ID.",
        )
        argparser.add_argument(
            "--vertex-location",
            type=str,
            required=True,
            help="Google Cloud Location.",
        )
        argparser.add_argument(
            "--vertex-experiment-name",
            type=str,
            required=True,
            help="Name of the Vertex AI Experiment.",
        )
        return argparser

    def _build(self) -> ExperimentTracker:
        """
        Initializes the Vertex AI Experiment and starts a new run.
        'run_name' is passed in from main.py (e.g., from an env var).
        """
        print(
            f"Initializing Vertex AI Experiment: {self._cli_args['vertex_experiment_name']}/{self.RUN_NAME}"
        )

        aiplatform.init(
            project=self._cli_args["vertex_project-id"],
            location=self._cli_args["vertex-location"],
            experiment=self._cli_args["vertex-experiment-name"],
        )

        # Start the run using the provided run_name
        aiplatform.start_run(run=self.RUN_NAME)

        return VertexAIExperimentTracker()
