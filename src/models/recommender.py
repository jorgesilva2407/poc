from torch import Module, Tensor
from abc import ABC, abstractmethod


class Recommender(ABC, Module):
    name: str

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @property
    @abstractmethod
    def hparams(self) -> dict[str, int | float | str]:
        pass

    @abstractmethod
    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """
        Compute the predicted scores for given user and item IDs.

        Args:
            user_ids (torch.Tensor): Tensor of shape (n,)
                User IDs.
            item_ids (torch.Tensor): Tensor of shape (n,)
                Item IDs.

        Returns:
            torch.Tensor: Tensor of shape (n,)
                Predicted scores for the user-item pairs.
        """
        pass
