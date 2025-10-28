"""Abstract base class for recommendation datasets."""

from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

TripletSample = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class RecommendationDataset(Dataset[TripletSample], ABC):
    """
    Abstract base class for recommendation datasets.
    Each sample consists of a user, a positive item, and multiple negative items.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of users in the dataset."""

    @abstractmethod
    def __getitem__(self, idx: int) -> TripletSample:
        """
        Retrieves the sample at the specified index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            TripletSample: A tuple containing user tensor, positive item tensor, and
                negative items tensor.
        """
