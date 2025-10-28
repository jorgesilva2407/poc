"""
Biased Singular Value Decomposition (Biased SVD) recommender model.
"""

from argparse import ArgumentParser
from torch import Tensor
from torch.nn import Embedding

from src.models.recommender import Recommender, RecommenderBuilder


class BiasedSVD(Recommender):
    """
    Biased Singular Value Decomposition (Biased SVD) recommender model.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__("BiasedSVD", num_users, num_items)
        self.embedding_dim = embedding_dim
        self.user_embedding = Embedding(num_users, embedding_dim)
        self.item_embedding = Embedding(num_items, embedding_dim)
        self.user_bias = Embedding(num_users, 1)
        self.item_bias = Embedding(num_items, 1)

    @property
    def hparams(self) -> dict[str, int]:
        return {
            "emb_dim": self.embedding_dim,
        }

    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()

        dot_product = (user_embeds * item_embeds).sum(dim=1)
        prediction = dot_product + user_b + item_b

        return prediction


class BiasedSVDBuilder(RecommenderBuilder):
    """
    Builder class for the Biased SVD recommender model.
    """

    @property
    def argparser(self) -> ArgumentParser:
        parser = super().argparser
        parser.add_argument(
            "--embedding-dim",
            type=int,
            required=True,
            help="Dimensionality of the user and item embeddings.",
        )
        return parser

    def _build(self, args: dict) -> BiasedSVD:
        return BiasedSVD(
            num_users=self._num_users,
            num_items=self._num_items,
            embedding_dim=args["embedding_dim"],
        )
