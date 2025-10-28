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


class RecommenderBuilder(ABC):
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

    def with_num_users_items(
        self, num_users: int, num_items: int
    ) -> "RecommenderBuilder":
        """
        Returns a new RecommenderBuilder with specified number of users and items.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.

        Returns:
            RecommenderBuilder: A new instance of RecommenderBuilder with updated parameters.
        """
        self._num_users = num_users
        self._num_items = num_items
        return self

    def build(self, args: dict) -> Recommender:
        """
        Build and return a recommender model instance.

        Args:
            args (dict): A dictionary of arguments for building the model.

        Returns:
            Recommender: An instance of the recommender model.
        """
        self._validate()
        return self._build(args)

    def _validate(self):
        """
        Validate the builder's configuration.
        """
        if not self._num_users or not self._num_items:
            raise ValueError(
                "Number of users and items must be set before building the model."
            )

    @abstractmethod
    def _build(self, args: dict) -> Recommender:
        """
        Internal method to build the recommender model after validation.

        Args:
            args (dict): A dictionary of arguments for building the model.

        Returns:
            Recommender: An instance of the recommender model.
        """
