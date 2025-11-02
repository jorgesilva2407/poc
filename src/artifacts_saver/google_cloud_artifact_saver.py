"""
ArtifactsSaver implementation that saves artifacts to Google Cloud Storage.
"""

import json
from pathlib import Path

import torch
from google.cloud import storage

from src.models.recommender import Recommender
from src.artifacts_saver.artifacts_saver import ArtifactsSaver, ArtifactsSaverBuilder


class GoogleCloudArtifactSaver(ArtifactsSaver):
    """
    ArtifactsSaver implementation that saves artifacts to Google Cloud Storage.
    """

    def __init__(self, bucket, gcloud_artifacts_path, local_artifacts_path):
        self.bucket = bucket
        self.gcloud_artifacts_path = gcloud_artifacts_path
        self.local_artifacts_path = local_artifacts_path
        self.local_artifacts_path.mkdir(parents=True, exist_ok=True)

    def _save_model(self, model: Recommender) -> None:
        local_path = self.local_artifacts_path / "model_weights.pth"
        gcloud_path = self.gcloud_artifacts_path / "model_weights.pth"
        torch.save(model.state_dict(), local_path)
        self._send_to_bucket(local_path, gcloud_path)

    def _save_metrics(
        self,
        hparams: dict[str, int | float | str],
        loss: float,
        metrics: dict[str, float],
    ) -> None:
        result = {}
        result["hparams"] = hparams
        result["loss"] = loss
        result["metrics"] = metrics
        local_path = self.local_artifacts_path / "metrics.json"
        gcloud_path = self.gcloud_artifacts_path / "metrics.json"
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(result, f)
        self._send_to_bucket(local_path, gcloud_path)

    def _save_user_metrics(self, user_metrics: dict[str, torch.Tensor]) -> None:
        local_metrics_path = self.local_artifacts_path / "user_metrics"
        local_metrics_path.mkdir(parents=True, exist_ok=True)
        gcloud_metrics_path = self.gcloud_artifacts_path / "user_metrics"
        for metric_name, metric_values in user_metrics.items():
            local_path = local_metrics_path / f"{metric_name}.pth"
            gcloud_path = gcloud_metrics_path / f"{metric_name}.pth"
            torch.save(metric_values, local_path)
            self._send_to_bucket(local_path, gcloud_path)

    def _send_to_bucket(self, local_path: Path, gcloud_path: Path) -> None:
        blob = self.bucket.blob(str(gcloud_path))
        blob.upload_from_filename(local_path)


class GoogleCloudArtifactSaverBuilder(ArtifactsSaverBuilder):
    """
    Builder for GoogleCloudArtifactSaver instances.
    """

    @property
    def argparser(self):
        parser = super().argparser
        parser.add_argument(
            "--gcs-bucket-name",
            type=str,
            required=True,
            help="Google Cloud Storage bucket name for saving artifacts.",
        )
        parser.add_argument(
            "--gcs-blob-base-path",
            type=str,
            required=True,
            help="Base path in the GCS bucket for saving artifacts.",
        )
        parser.add_argument(
            "--temp-local-path",
            type=str,
            default="/tmp/artifacts",
            help="Temporary local path for storing artifacts before uploading to GCS.",
        )
        return parser

    def _build(self, model_id: str) -> ArtifactsSaver:
        gcp_bucket_name = self._cli_args["gcs_bucket_name"]
        gcp_blob_base_path = self._cli_args["gcs_blob_base_path"]
        temp_local_path = self._cli_args["temp_local_path"]

        return GoogleCloudArtifactSaver(
            bucket=storage.Client().bucket(gcp_bucket_name),
            gcloud_artifacts_path=Path(gcp_blob_base_path) / model_id,
            local_artifacts_path=Path(temp_local_path) / model_id,
        )
