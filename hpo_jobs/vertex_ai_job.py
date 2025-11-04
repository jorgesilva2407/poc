"""Hyperparameter Optimization Job Class for Google Cloud AI Platform."""

import os
from enum import StrEnum
from datetime import datetime

from google.cloud import aiplatform
from google.oauth2 import service_account
from google.cloud.aiplatform import hyperparameter_tuning as hpt

ParamSpecType = (
    hpt.DoubleParameterSpec
    | hpt.DiscreteParameterSpec
    | hpt.IntegerParameterSpec
    | hpt.CategoricalParameterSpec
)


class HPOJob:
    """Hyperparameter Optimization Job for Google Cloud AI Platform."""

    class Environment(StrEnum):
        """Environment variable names."""

        PROJECT_ID = "PROJECT_ID"
        REGION = "REGION"
        GOOGLE_APPLICATION_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"
        IMAGE_URI = "IMAGE_URI"
        BUCKET = "BUCKET"

    MAX_TRIAL_COUNT: int = 20
    PARALLEL_TRIAL_COUNT: int = 4
    LOGGER_NAME = "TensorBoard"
    ARTIFACT_SAVER = "GoogleCloud"
    BLOB_BASE_PATH = "artifacts/hpo_jobs"
    EXPERIMENT_TRACKER = "VertexAI"
    METRIC_SPEC = {"NDCG@5": "maximize"}
    DEFAULT_PARAM_SPEC = {
        "batch_size": hpt.DiscreteParameterSpec(values=[64, 128, 256], scale="linear"),
        "learning_rate": hpt.DoubleParameterSpec(min=1e-5, max=1e-2, scale="log"),
        "weight_decay": hpt.DoubleParameterSpec(min=1e-6, max=1e-2, scale="log"),
    }

    project_id = os.getenv(Environment.PROJECT_ID.value)
    region = os.getenv(Environment.REGION.value)
    service_account_file = os.getenv(Environment.GOOGLE_APPLICATION_CREDENTIALS.value)
    image_uri = os.getenv(Environment.IMAGE_URI.value)
    bucket_name = os.getenv(Environment.BUCKET.value)
    bucket_uri = f"gs://{os.getenv(Environment.BUCKET.value)}"

    model_name: str
    display_name: str
    experiment_name: str
    custom_job: aiplatform.CustomContainerTrainingJob
    all_interactions_csv_path: str
    train_interactions_csv_path: str
    val_interactions_csv_path: str
    test_interactions_csv_path: str
    model_specific_args: dict[str, str]

    def __init__(
        self,
        model_name: str,
        all_interactions_csv_path: str,
        train_interactions_csv_path: str,
        val_interactions_csv_path: str,
        test_interactions_csv_path: str,
        model_specific_args: dict[str, any] | None = None,
    ):
        missing_env_vars = [
            var.value for var in self.Environment if os.getenv(var.value) is None
        ]
        if missing_env_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_env_vars)}"
            )

        self.model_name = model_name
        self.display_name = f"{model_name}HPO"
        self.experiment_name = (
            f"{model_name.lower()}hpo{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        self.all_interactions_csv_path = all_interactions_csv_path
        self.train_interactions_csv_path = train_interactions_csv_path
        self.val_interactions_csv_path = val_interactions_csv_path
        self.test_interactions_csv_path = test_interactions_csv_path
        self.model_specific_args = model_specific_args or dict()

        credentials = service_account.Credentials.from_service_account_file(
            self.service_account_file
        )
        aiplatform.init(
            project=self.project_id,
            location=self.region,
            experiment=self.experiment_name,
            experiment_description=f"Hyperparameter optimization for {model_name} model",
            staging_bucket=self.bucket_uri,
            credentials=credentials,
        )

        worker_pool_specs = self._get_worker_pool_specs()
        self.custom_job = aiplatform.CustomJob(
            display_name=self.display_name,
            worker_pool_specs=[worker_pool_specs],
        )

    def run(self, parameter_spec: dict[str, ParamSpecType]):
        """Run the Hyperparameter Optimization Job."""
        spec = self._get_param_spec(parameter_spec)

        tuning_job = aiplatform.HyperparameterTuningJob(
            display_name=self.display_name,
            custom_job=self.custom_job,
            metric_spec=self.METRIC_SPEC,
            parameter_spec=spec,
            max_trial_count=self.MAX_TRIAL_COUNT,
            parallel_trial_count=self.PARALLEL_TRIAL_COUNT,
        )

        tuning_job.run()

    def _get_args(self):
        base_args = {
            "--model": self.model_name,
            "--logger": self.LOGGER_NAME,
            "--artifact-saver": self.ARTIFACT_SAVER,
            "--gcs-bucket-name": self.bucket_name,
            "--gcs-blob-base-path": self.BLOB_BASE_PATH,
            "--experiment-tracker": self.EXPERIMENT_TRACKER,
            "--vertex-project-id": self.project_id,
            "--vertex-location": self.region,
            "--vertex-experiment-name": self.experiment_name,
            "--all-interactions-csv": self.all_interactions_csv_path,
            "--train-interactions-csv": self.train_interactions_csv_path,
            "--validation-interactions-csv": self.val_interactions_csv_path,
            "--test-interactions-csv": self.test_interactions_csv_path,
        }
        base_args.update(self.model_specific_args)
        return base_args

    def _get_worker_pool_specs(self):
        """Get the worker pool specifications for the custom job."""
        args = self._get_args()
        args_list = [str(item) for pair in args.items() for item in pair]
        worker_pool_specs = {
            "replica_count": 1,
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "container_spec": {
                "image_uri": self.image_uri,
                "command": [],
                "args": args_list,
            },
        }
        return worker_pool_specs

    def _get_param_spec(
        self, parameter_spec: dict[str, ParamSpecType]
    ) -> dict[str, ParamSpecType]:
        spec = self.DEFAULT_PARAM_SPEC.copy()
        spec.update(parameter_spec)
        return spec


def main():
    """Example usage of HPOJob."""
    hpo_job = HPOJob(
        model_name="ExampleModel",
        all_interactions_csv_path="gs://your-bucket/path/to/all_interactions.csv",
        train_interactions_csv_path="gs://your-bucket/path/to/train_interactions.csv",
        val_interactions_csv_path="gs://your-bucket/path/to/val_interactions.csv",
        test_interactions_csv_path="gs://your-bucket/path/to/test_interactions.csv",
    )

    parameter_spec = HPOJob.DEFAULT_PARAM_SPEC

    hpo_job.run(parameter_spec=parameter_spec)


if __name__ == "__main__":
    main()
