"""Local filesystem implementation of ArtifactsSaver."""

import json
from pathlib import Path

import torch

from src.models.recommender import Recommender
from src.artifacts_saver.artifacts_saver import ArtifactsSaver, ArtifactsSaverBuilder


class LocalArtifactsSaver(ArtifactsSaver):
    """
    ArtifactsSaver implementation that saves artifacts to the local filesystem.
    """

    def __init__(self, local_artifacts_path):
        self.local_artifacts_path = local_artifacts_path
        self.local_artifacts_path.mkdir(parents=True, exist_ok=True)

    def _save_model(self, model: Recommender) -> None:
        local_path = self.local_artifacts_path / "model_weights.pth"
        torch.save(model.state_dict(), local_path)

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
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(result, f)

    def _save_user_metrics(self, user_metrics: dict[str, torch.Tensor]) -> None:
        local_metrics_path = self.local_artifacts_path / "user_metrics"
        local_metrics_path.mkdir(parents=True, exist_ok=True)
        for metric_name, metric_values in user_metrics.items():
            local_path = local_metrics_path / f"{metric_name}.pth"
            torch.save(metric_values, local_path)


class LocalArtifactsSaverBuilder(ArtifactsSaverBuilder):
    """
    Builder for LocalArtifactsSaver instances.
    """

    @property
    def argparser(self):
        parser = super().argparser
        parser.add_argument(
            "--local-artifacts-path",
            type=str,
            required=True,
            help="Path to save local artifacts.",
        )
        return parser

    def _build(self, model_id: str) -> ArtifactsSaver:
        local_artifacts_path = Path(self._cli_args["local_artifacts_path"]) / model_id
        return LocalArtifactsSaver(local_artifacts_path)
