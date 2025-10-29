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

    @abstractmethod
    def save_artifacts(
        self,
        hparams: dict[str, int | float | str],
        model: Recommender,
        loss: float,
        metrics: dict[str, float],
        user_metrics: dict[str, Tensor],
    ) -> None:
        """Save model artifacts such as weights and evaluation metrics."""


class ArtifactsSaverBuilder(ConfigurableBuilder[ArtifactsSaver], ABC):
    """
    Abstract base class for building ArtifactsSaver instances.
    """
