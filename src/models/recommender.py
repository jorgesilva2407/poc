"""
Abstract base class for recommender models.
"""

from argparse import ArgumentParser
from abc import ABC, abstractmethod
from typing import NamedTuple
from torch import Tensor
import torch
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
    batch_size: int = 256

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

    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """
        Compute the predicted scores for given user and item IDs.
        Automatically batches predictions if input size exceeds batch_size.

        Args:
            user_ids (torch.Tensor): Tensor of shape (n,)
                User IDs.
            item_ids (torch.Tensor): Tensor of shape (n,)
                Item IDs.

        Returns:
            torch.Tensor: Tensor of shape (n,)
                Predicted scores for the user-item pairs.
        """
        # If input is smaller than batch size, process normally
        if len(user_ids) <= self.batch_size:
            return self._forward(user_ids, item_ids)

        # Otherwise, process in batches
        predictions = []
        for i in range(0, len(user_ids), self.batch_size):
            batch_user_ids = user_ids[i : i + self.batch_size]
            batch_item_ids = item_ids[i : i + self.batch_size]
            batch_predictions = self._forward(batch_user_ids, batch_item_ids)
            predictions.append(batch_predictions)

        return torch.cat(predictions, dim=0)

    @abstractmethod
    def _forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
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
