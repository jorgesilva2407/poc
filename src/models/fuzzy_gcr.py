"""Fuzzy Graph Collaborative Reasoning (FuzzGCR) implementation."""

from argparse import ArgumentParser
from typing import Literal

import torch
import pandas as pd

from src.models.recommender import Recommender, Context
from src.models.base_gcr import BaseGCR, BaseGCRFactory
from src.models.encoders.lookup_encoder import LookupEncoder
from src.models.logical_modules.fuzzy import (
    FuzzyNOT,
    FuzzyProductOR,
    FuzzyGodelOR,
    FuzzyLukasiewiczOR,
    FuzzySmoothOR,
)

LogicSystem = Literal["product", "godel", "lukasiewicz", "smooth"]


class FuzzyGCR(BaseGCR):
    """Graph Collaborative Reasoning with Fuzzy Logic operators."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        interactions: pd.DataFrame,
        user_item_embedding_dim: int,
        event_embedding_dim: int,
        hidden_dim: int,
        num_neighbors: int,
        logic_system: LogicSystem,
        dropout_rate: float,
        stiffness: float | None,
    ):
        # 1. Create User and Item Encoders
        user_encoder = LookupEncoder(num_users, user_item_embedding_dim)
        item_encoder = LookupEncoder(num_items, user_item_embedding_dim)

        # 2. Select Fuzzy Logic System
        not_module = FuzzyNOT()

        if logic_system == "product":
            or_module = FuzzyProductOR()
        elif logic_system == "godel":
            or_module = FuzzyGodelOR()
        elif logic_system == "lukasiewicz":
            or_module = FuzzyLukasiewiczOR()
        elif logic_system == "smooth":
            if stiffness is None:
                raise ValueError(
                    "Stiffness parameter must be provided for Smooth Fuzzy Logic."
                )
            or_module = FuzzySmoothOR(stiffness)
        else:
            raise ValueError(f"Unknown logic system: {logic_system}")

        self.user_item_embedding_dim = user_item_embedding_dim
        self.logic_system = logic_system
        self.stiffness = stiffness

        # 3. Initialize Base GCR
        super().__init__(
            name=f"FuzzGCR-{logic_system}",
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
            should_permute=False,
            dropout_rate=dropout_rate,
        )

    def _encode_event(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes (User, Item) pairs into the Fuzzy Logic Space [0, 1]^d.
        Overridden to use Sigmoid instead of Normalize.
        """
        raw = self._likes(input_tensor)
        return torch.sigmoid(raw)

    def _init_true_anchor(self, dim: int) -> torch.Tensor:
        """
        Initialize the TRUE anchor vector.
        In Fuzzy Logic, Absolute Truth is represented by 1.0.
        """
        return torch.ones(dim)

    def sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity using Dot Product.
        """
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)

        # Element-wise multiplication then sum (Dot product)
        return (a * b).sum(dim=-1)

    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Fuzzy Logic operators satisfy axiomatic rules by definition.
        No explicit logical regularization is required.
        """
        # Clear cache to prevent memory leaks, even if unused
        self._cached_event_embeddings = None
        return torch.tensor(0.0, device=self.TRUE.device)

    @property
    def inverse_temperature(self) -> float:
        # FuzzCR uses a larger scaling coefficient (gamma), e.g., 20-30[cite: 1043].
        return 20.0

    @property
    def hparams(self):
        model_hparams = {
            "ui_emb_dim": self.user_item_embedding_dim,
        }
        if self.stiffness is not None:
            model_hparams["stiffness"] = self.stiffness
        super_hparams = super().hparams
        return {**super_hparams, **model_hparams}


class FuzzyGCRFactory(BaseGCRFactory):
    """Factory for creating FuzzyGCR model instances."""

    @property
    def argparser(self) -> ArgumentParser:
        parser = super().argparser
        parser.add_argument(
            "--user-item-embedding-dim",
            type=int,
            required=True,
            help="Dimension of user and item embeddings.",
        )
        return parser

    def _create(self, logic_system: str, context: Context, args: dict) -> Recommender:
        user_item_embedding_dim = args["user_item_embedding_dim"]
        event_embedding_dim = args["event_embedding_dim"]
        hidden_dim = args["hidden_dim"]
        num_neighbors = args["num_neighbors"]
        dropout_rate = args["dropout_rate"]
        stiffness = args.get("stiffness", None)
        return FuzzyGCR(
            num_users=context.num_users,
            num_items=context.num_items,
            interactions=context.interactions,
            user_item_embedding_dim=user_item_embedding_dim,
            event_embedding_dim=event_embedding_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            logic_system=logic_system,
            dropout_rate=dropout_rate,
            stiffness=stiffness,
        )


class FuzzyProductGCRFactory(FuzzyGCRFactory):
    """Factory for creating FuzzyGCR model instances with Product logic."""

    def create(self, context: Context, args: dict) -> Recommender:
        return self._create("product", context, args)


class FuzzyGodelGCRFactory(FuzzyGCRFactory):
    """Factory for creating FuzzyGCR model instances with Godel logic."""

    def create(self, context: Context, args: dict) -> Recommender:
        return self._create("godel", context, args)


class FuzzyLukasiewiczGCRFactory(FuzzyGCRFactory):
    """Factory for creating FuzzyGCR model instances with Lukasiewicz logic."""

    def create(self, context: Context, args: dict) -> Recommender:
        return self._create("lukasiewicz", context, args)


class FuzzySmoothGCRFactory(FuzzyGCRFactory):
    """Factory for creating FuzzyGCR model instances with Smooth logic."""

    @property
    def argparser(self) -> ArgumentParser:
        parser = super().argparser
        parser.add_argument(
            "--stiffness",
            type=float,
            required=True,
            help="Stiffness parameter for Smooth Fuzzy Logic.",
        )
        return parser

    def create(self, context: Context, args: dict) -> Recommender:
        return self._create("smooth", context, args)
