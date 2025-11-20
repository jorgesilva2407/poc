"""Graph Collaborative Reasoning (GCR) model implementation."""

from argparse import ArgumentParser

import pandas as pd

from src.models.recommender import Recommender, Context
from src.models.base_gcr import BaseGCR, BaseGCRFactory
from src.models.encoders.lookup_encoder import LookupEncoder
from src.models.logical_modules.neural_not import NeuralNOT
from src.models.logical_modules.neural_or import NeuralOR
from src.models.logical_modules.regularizers import (
    make_not_hook_factory,
    make_or_hook_factory,
)


class GCR(BaseGCR):
    """Graph Collaborative Reasoning (GCR) model."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        interactions: pd.DataFrame,
        user_item_embedding_dim: int,
        event_embedding_dim: int,
        hidden_dim: int,
        num_neighbors: int,
        reg_weight: float,
    ):
        # 1. Create User and Item Encoders
        user_encoder = LookupEncoder(num_users, user_item_embedding_dim)
        item_encoder = LookupEncoder(num_items, user_item_embedding_dim)

        # 2. Create Neural Logic Components
        or_module = NeuralOR(event_embedding_dim, hidden_dim)
        not_module = NeuralNOT(event_embedding_dim, hidden_dim)

        # Store regularization weight
        self.user_item_embedding_dim = user_item_embedding_dim
        self.reg_weight = reg_weight

        # 3. Initialize Base GCR
        super().__init__(
            name="GCR",
            num_users=num_users,
            num_items=num_items,
            interactions=interactions,
            user_encoder=user_encoder,
            item_encoder=item_encoder,
            or_module=or_module,
            not_module=not_module,
            event_embedding_dim=event_embedding_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            should_permute=True,
        )

        # 4. Register Regularization Hooks
        self.register_regularizers_hooks()

    def register_regularizers_hooks(self):
        """Register gradient hooks for logical modules."""
        # Requires self.reg_weight to be defined in this class

        not_factory = make_not_hook_factory(
            get_event_embeddings=lambda: self._cached_event_embeddings,
            NOT=self._not,
            sim=self.sim,
            reg_weight=self.reg_weight,
        )

        or_factory = make_or_hook_factory(
            get_event_embeddings=lambda: self._cached_event_embeddings,
            TRUE=self.TRUE,
            OR=self._or,
            NOT=self._not,
            sim=self.sim,
            reg_weight=self.reg_weight,
        )

        for p in self._not.parameters():
            p.register_hook(not_factory(p))

        for p in self._or.parameters():
            p.register_hook(or_factory(p))

    @property
    def hparams(self) -> dict[str, int]:
        model_hparams = {
            "ui_emb_dim": self.user_item_embedding_dim,
            "reg_weight": self.reg_weight,
        }
        super_hparams = super().hparams
        return {**super_hparams, **model_hparams}


class GCRFactory(BaseGCRFactory):
    """Factory for creating GCR model instances."""

    @property
    def argparser(self) -> ArgumentParser:
        parser = super().argparser
        parser.add_argument(
            "--reg_weight",
            type=float,
            default=0.01,
            help="Regularization weight for logical modules.",
        )
        parser.add_argument(
            "--user_item_embedding_dim",
            type=int,
            default=32,
            help="Dimension of user and item embeddings.",
        )
        return parser

    def create(self, context: Context, args: dict) -> Recommender:
        event_embedding_dim = args["event_embedding_dim"]
        hidden_dim = args["hidden_dim"]
        num_neighbors = args["num_neighbors"]
        reg_weight = args["reg_weight"]
        user_item_embedding_dim = args["user_item_embedding_dim"]
        return GCR(
            num_users=context.num_users,
            num_items=context.num_items,
            interactions=context.interactions,
            user_item_embedding_dim=user_item_embedding_dim,
            event_embedding_dim=event_embedding_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            reg_weight=reg_weight,
        )
