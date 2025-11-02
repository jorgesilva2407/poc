"""
Abstract base class for recommender models.
"""

from argparse import ArgumentParser
from abc import ABC, abstractmethod
from typing import NamedTuple
from torch import Tensor
from torch.nn import Module
from pandas import DataFrame


class Context(NamedTuple):
    """
    Context information for the recommender model.
    """

    num_users: int
    num_items: int
    interactions_df: DataFrame


class Recommender(ABC, Module):
    """
    Abstract base class for recommender models.
    """

    name: str
    num_users: int
    num_items: int

    def __init__(self, name: str, num_users: int, num_items: int) -> None:
        super().__init__()
        self.name = name
        self.num_users = num_users
        self.num_items = num_items

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


class RecommenderFactory(ABC):
    """
    Abstract base class for recommender model builders.
    """

    _num_users: int
    _num_items: int

    @property
    def argparser(self) -> ArgumentParser:
        """
        Returns an ArgumentParser for building the recommender model.

        Returns:
            ArgumentParser: An argument parser for building the model.
        """
        return ArgumentParser(add_help=False)

    @abstractmethod
    def create(self, context: Context, args: dict) -> Recommender:
        """
        Build and return a recommender model instance.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            args (dict): A dictionary of arguments for building the model.

        Returns:
            Recommender: An instance of the recommender model.
        """
