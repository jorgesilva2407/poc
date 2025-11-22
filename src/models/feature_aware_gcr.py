"""Feature-Aware Graph Collaborative Reasoning (GCR) model implementation."""

import json
from argparse import ArgumentParser
from typing import Literal

import torch
import torch.nn.functional as F
import pandas as pd

from src.models.recommender import Recommender, Context
from src.models.base_gcr import BaseGCR, BaseGCRFactory
from src.models.encoders.feature_aware_encoder import FeatureAwareEncoder
from src.models.logical_modules.neural import NeuralOR, NeuralNOT
from src.models.logical_modules.regularizers import or_regularizer, not_regularizer


class FeatureAwareGCR(BaseGCR):
    """Feature-Aware Graph Collaborative Reasoning (GCR) model."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        interactions: pd.DataFrame,
        user_feature_csv: str,
        item_feature_csv: str,
        feature_metadata: dict[
            Literal["user", "item"],
            dict[str, Literal["boolean", "numerical", "categorical"]],
        ],
        categorical_embedding_dim: int,
        embed_id: bool,
        encoder_output_dim: int,
        event_embedding_dim: int,
        hidden_dim: int,
        num_neighbors: int,
        reg_weight: float,
        dropout_rate: float,
    ):
        """
        Initialize Feature-Aware GCR model.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            interactions (pd.DataFrame): User-item interactions.
            user_feature_csv (str): Path to user features CSV.
            item_feature_csv (str): Path to item features CSV.
            feature_metadata (dict): Feature metadata for users and items.
            categorical_embedding_dim (int): Dimension for categorical embeddings.
            embed_id (bool): Whether to embed entity IDs.
            encoder_output_dim (int): Output dimension of encoders.
            event_embedding_dim (int): Dimension of event embeddings.
            hidden_dim (int): Hidden dimension for logical modules.
            num_neighbors (int): Number of neighbors to sample.
            reg_weight (float): Regularization weight.
            dropout_rate (float): Dropout rate.
        """
        # 1. Create User and Item Encoders with Feature Awareness
        user_encoder = FeatureAwareEncoder(
            num_entities=num_users,
            feature_path=user_feature_csv,
            feature_metadata=feature_metadata["user"],
            categorical_embedding_dim=categorical_embedding_dim,
            embed_id=embed_id,
            output_dim=encoder_output_dim,
            dropout_rate=dropout_rate,
        )

        item_encoder = FeatureAwareEncoder(
            num_entities=num_items,
            feature_path=item_feature_csv,
            feature_metadata=feature_metadata["item"],
            categorical_embedding_dim=categorical_embedding_dim,
            embed_id=embed_id,
            output_dim=encoder_output_dim,
            dropout_rate=dropout_rate,
        )

        # 2. Create Neural Logic Components
        or_module = NeuralOR(event_embedding_dim, hidden_dim, dropout_rate)
        not_module = NeuralNOT(event_embedding_dim, hidden_dim, dropout_rate)

        # Store hyperparameters
        self.categorical_embedding_dim = categorical_embedding_dim
        self.embed_id = embed_id
        self.encoder_output_dim = encoder_output_dim
        self.reg_weight = reg_weight

        # 3. Initialize Base GCR
        super().__init__(
            name="FeatureAwareGCR",
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
            dropout_rate=dropout_rate,
        )

    @property
    def hparams(self) -> dict[str, int]:
        model_hparams = {
            "cat_emb_dim": self.categorical_embedding_dim,
            "embed_id": self.embed_id,
            "encoder_out_dim": self.encoder_output_dim,
            "reg_weight": self.reg_weight,
        }
        super_hparams = super().hparams
        return {**super_hparams, **model_hparams}

    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Computes the regularization loss for the logical modules based on
        the events cached during the last forward pass.
        """
        if self._cached_event_embeddings is None:
            return torch.tensor(0.0)

        # Use current cached events
        X = self._cached_event_embeddings

        # Compute NOT regularizer loss
        loss_not = not_regularizer(self._not, X, sim=self.sim)

        # Compute OR regularizer loss
        loss_or = or_regularizer(self._or, self._not, self.TRUE, X, sim=self.sim)

        # Clear cached events
        self._cached_event_embeddings = None

        return self.reg_weight * (loss_not + loss_or)

    def _init_true_anchor(self, dim: int) -> torch.Tensor:
        """Initialize the TRUE anchor vector."""
        raw_anchor = torch.randn(dim)
        return F.normalize(raw_anchor, p=2, dim=0)

    @property
    def inverse_temperature(self) -> float:
        """
        Returns the inverse temperature parameter for the Feature-Aware GCR model.

        Returns:
            float: The inverse temperature.
        """
        return 10.0


class FeatureAwareGCRFactory(BaseGCRFactory):
    """Factory for creating Feature-Aware GCR model instances."""

    @property
    def argparser(self) -> ArgumentParser:
        parser = super().argparser
        parser.add_argument(
            "--reg-weight",
            type=float,
            required=True,
            help="Regularization weight for logical modules.",
        )
        parser.add_argument(
            "--categorical-embedding-dim",
            type=int,
            required=True,
            help="Dimension of categorical feature embeddings.",
        )
        parser.add_argument(
            "--embed-id",
            action="store_true",
            help="Whether to learn embeddings for entity IDs.",
        )
        parser.add_argument(
            "--encoder-output-dim",
            type=int,
            required=True,
            help="Output dimension of feature-aware encoders.",
        )
        parser.add_argument(
            "--user-feature-csv",
            type=str,
            required=True,
            help="Path to user features CSV file.",
        )
        parser.add_argument(
            "--item-feature-csv",
            type=str,
            required=True,
            help="Path to item features CSV file.",
        )
        parser.add_argument(
            "--feature-metadata-json",
            type=str,
            required=True,
            help="Path to JSON file containing feature metadata for users and items.",
        )
        return parser

    def create(self, context: Context, args: dict) -> Recommender:
        # Load feature metadata from JSON file
        with open(args["feature_metadata_json"], "r", encoding="utf-8") as f:
            feature_metadata = json.load(f)

        event_embedding_dim = args["event_embedding_dim"]
        hidden_dim = args["hidden_dim"]
        num_neighbors = args["num_neighbors"]
        dropout_rate = args["dropout_rate"]
        reg_weight = args["reg_weight"]
        categorical_embedding_dim = args["categorical_embedding_dim"]
        embed_id = args["embed_id"]
        encoder_output_dim = args["encoder_output_dim"]
        user_feature_csv = args["user_feature_csv"]
        item_feature_csv = args["item_feature_csv"]

        return FeatureAwareGCR(
            num_users=context.num_users,
            num_items=context.num_items,
            interactions=context.interactions,
            user_feature_csv=user_feature_csv,
            item_feature_csv=item_feature_csv,
            feature_metadata=feature_metadata,
            categorical_embedding_dim=categorical_embedding_dim,
            embed_id=embed_id,
            encoder_output_dim=encoder_output_dim,
            event_embedding_dim=event_embedding_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            reg_weight=reg_weight,
            dropout_rate=dropout_rate,
        )
