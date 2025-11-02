"""
Abstract base class for saving model artifacts such as weights and evaluation metrics.
"""

from abc import ABC, abstractmethod

from torch import Tensor

from src.models.recommender import Recommender
from src.configurable_builder import ConfigurableBuilder


class ArtifactsSaver(ABC):
    """
    Abstract base class for saving model artifacts such as weights and evaluation metrics.
    """

    def save_artifacts(
        self,
        hparams: dict[str, int | float | str],
        model: Recommender,
        loss: float,
        metrics: dict[str, float],
        user_metrics: dict[str, Tensor],
    ) -> None:
        """Save model artifacts such as weights and evaluation metrics."""
        self._save_model(model)
        self._save_metrics(hparams, loss, metrics)
        self._save_user_metrics(user_metrics)

    @abstractmethod
    def _save_model(self, model: Recommender) -> None:
        """Save the model weights."""

    @abstractmethod
    def _save_metrics(
        self,
        hparams: dict[str, int | float | str],
        loss: float,
        metrics: dict[str, float],
    ) -> None:
        """Save the evaluation metrics."""

    @abstractmethod
    def _save_user_metrics(self, user_metrics: dict[str, Tensor]) -> None:
        """Save user-specific metrics."""


class ArtifactsSaverBuilder(ConfigurableBuilder[ArtifactsSaver], ABC):
    """
    Abstract base class for building ArtifactsSaver instances.
    """
