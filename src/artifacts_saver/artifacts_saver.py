from torch import Tensor
from abc import ABC, abstractmethod
from models.recommender import Recommender


class ArtifactsSaver(ABC):
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
        pass


class ArtifactsSaverBuilder(ABC):
    @abstractmethod
    def build(self, model_id: str) -> ArtifactsSaver:
        """Build and return an ArtifactsSaver instance."""
        pass
