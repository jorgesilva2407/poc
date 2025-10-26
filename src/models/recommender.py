"""
Abstract base class for recommender models.
"""

from argparse import ArgumentParser
from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module


class Recommender(ABC, Module):
    """
    Abstract base class for recommender models.
    """

    name: str

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @property
    @abstractmethod
    def hparams(self) -> dict[str, int | float | str]:
        """
        Returns the hyperparameters of the recommender model.

        Returns:
            dict[str, int | float | str]: A dictionary containing the hyperparameters.
        """

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


class RecommenderBuilder(ABC):
    """
    Abstract base class for recommender model builders.
    """

    @property
    def argparser(self) -> ArgumentParser:
        """
        Returns an ArgumentParser for building the recommender model.

        Returns:
            ArgumentParser: An argument parser for building the model.
        """
        parser = ArgumentParser(add_help=False)
        parser.add_argument(
            "--num-users",
            type=int,
            required=True,
            help="Number of users in the dataset.",
        )
        parser.add_argument(
            "--num-items",
            type=int,
            required=True,
            help="Number of items in the dataset.",
        )
        return parser

    @abstractmethod
    def build(self, args: dict) -> Recommender:
        """
        Build and return a recommender model instance.

        Args:
            args (dict): A dictionary of arguments for building the model.

        Returns:
            Recommender: An instance of the recommender model.
        """
