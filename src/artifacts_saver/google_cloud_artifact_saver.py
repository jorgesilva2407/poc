import json
import torch
from pathlib import Path
from google.cloud import storage
from models.recommender import Recommender
from src.artifacts_saver.artifacts_saver import ArtifactsSaver, ArtifactsSaverBuilder


class GoogleCloudArtifactSaver(ArtifactsSaver):
    def __init__(self, bucket, gcloud_artifacts_path, local_artifacts_path):
        self.bucket = bucket
        self.gcloud_artifacts_path = gcloud_artifacts_path
        self.local_artifacts_path = local_artifacts_path
        self.local_artifacts_path.mkdir(parents=True, exist_ok=True)

    def save_artifacts(
        self,
        hparams: dict[str, int | float | str],
        model: Recommender,
        loss: float,
        metrics: dict[str, float],
        user_metrics: dict[str, torch.Tensor],
    ) -> None:
        self._save_model(model)
        self._save_metrics(hparams, loss, metrics)
        self._save_user_metrics(user_metrics)

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
        with open(local_path, "w") as f:
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
    def __init__(
        self, local_artifacts_path: Path, gcp_bucket_name: str, gcp_blob_base_path: Path
    ):
        super().__init__()
        self.local_artifacts_path = local_artifacts_path
        self.gcp_bucket_name = gcp_bucket_name
        self.gcp_blob_base_path = gcp_blob_base_path

    def build(self, model_id: str) -> ArtifactsSaver:
        return GoogleCloudArtifactSaver(
            bucket=storage.Client().bucket(self.gcp_bucket_name),
            gcloud_artifacts_path=self.gcp_blob_base_path / model_id,
            local_artifacts_path=self.local_artifacts_path / model_id,
        )
