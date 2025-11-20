"""Graph Collaborative Reasoning (GCR) model implementation."""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.recommender import Recommender, RecommenderFactory, Context
from src.models.logical_modules.neural_not import NeuralNOT
from src.models.logical_modules.neural_or import NeuralOR
from src.models.logical_modules.regularizers import (
    make_not_hook_factory,
    make_or_hook_factory,
)


class GCR(Recommender):
    """
    Graph Collaborative Reasoning (GCR) model.
    """

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
        super().__init__("GCR", num_users, num_items)
        self.num_neighbors = num_neighbors
        self.reg_weight = reg_weight

        # For logical regularizer to operate on event vectors
        # Filled in forward()
        self._cached_event_embeddings = None

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, user_item_embedding_dim)
        self.item_embedding = nn.Embedding(num_items, user_item_embedding_dim)

        # Precompute user and item interactions
        self.user_interactions = (
            interactions.groupby("user_id")["item_id"]
            .agg(lambda x: np.array(x.unique(), dtype=np.int64))
            .to_dict()
        )

        self.item_interactions = (
            interactions.groupby("item_id")["user_id"]
            .agg(lambda x: np.array(x.unique(), dtype=np.int64))
            .to_dict()
        )

        # TRUE anchor for logical operations
        self.register_buffer("TRUE", torch.rand(user_item_embedding_dim))

        # Logical operators
        self._or = NeuralOR(user_item_embedding_dim, hidden_dim)
        self._not = NeuralNOT(user_item_embedding_dim, hidden_dim)

        # Event composition module
        self._likes = nn.Sequential(
            nn.Linear(
                self.user_embedding.embedding_dim + self.item_embedding.embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, event_embedding_dim),
        )

        # Register gradient hooks for logical regularizers
        self.register_regularizers_hooks()

    def sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two tensors."""
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        # pylint: disable=not-callable
        return F.cosine_similarity(a, b, dim=-1)
        # pylint: enable=not-callable

    def register_regularizers_hooks(self):
        """Register gradient hooks for logical modules."""
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
        return {
            "ui_emb_dim": self.user_embedding.embedding_dim,
            "event_emb_dim": self._likes[-1].out_features,
            "hidden_dim": self._likes[0].out_features,
            "num_neighbors": self.num_neighbors,
            "reg_weight": self.reg_weight,
        }

    def _sample_neighbors(
        self,
        entity_ids: np.ndarray,
        exclusion_ids: np.ndarray,
        interactions_map: dict[int, np.ndarray],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples neighbors for a batch of entities, excluding specific IDs.
        Returns indices and a boolean mask. Padded indices are set to 0.
        """
        batch_size = len(entity_ids)
        neighbor_indices = []
        masks = []

        for idx in range(batch_size):
            eid = entity_ids[idx]
            exclude_id = exclusion_ids[idx]

            # Get interactions
            neighbors = interactions_map.get(eid, np.array([], dtype=np.int64))

            # Filter exclusion ID
            valid_neighbors = neighbors[neighbors != exclude_id]

            if len(valid_neighbors) > self.num_neighbors:
                # Sample without replacement
                sampled = np.random.choice(
                    valid_neighbors, self.num_neighbors, replace=False
                )
                # True for valid neighbors
                mask = np.ones(self.num_neighbors, dtype=bool)
            else:
                # Take all available, pad remainder with 0
                sampled = valid_neighbors
                pad_len = self.num_neighbors - len(sampled)

                # 1s for valid, 0s for padding
                mask = np.concatenate(
                    [
                        np.ones(len(sampled), dtype=bool),
                        np.zeros(pad_len, dtype=bool),
                    ]
                )
                # Use 0 for padding IDs
                sampled = np.concatenate([sampled, np.zeros(pad_len, dtype=np.int64)])

            neighbor_indices.append(sampled)
            masks.append(mask)

        return (
            torch.tensor(np.array(neighbor_indices), device=device, dtype=torch.long),
            torch.tensor(np.array(masks), device=device, dtype=torch.bool),
        )

    def _compute_neighbor_events(
        self,
        center_emb: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_embedding_layer: nn.Embedding,
        center_is_user: bool,
    ) -> torch.Tensor:
        """
        Computes event embeddings for neighbors.
        center_emb: (B, D)
        neighbor_indices: (B, NumNeighbors)
        """
        batch_size = center_emb.size(0)

        # Expand center embedding: (B, 1, D) -> (B, N, D)
        center_expanded = center_emb.unsqueeze(1).expand(-1, self.num_neighbors, -1)

        # Get neighbor embeddings: (B, N, D)
        # Padding indices are 0, which is a valid index.
        # The resulting embeddings for padded positions will be masked out later.
        neighbor_emb = neighbor_embedding_layer(neighbor_indices)

        # Concatenate based on order required by _likes (User, Item)
        if center_is_user:
            # Center is User, Neighbor is Item -> (User, NeighborItem)
            inp = torch.cat([center_expanded, neighbor_emb], dim=-1)
        else:
            # Center is Item, Neighbor is User -> (NeighborUser, Item)
            inp = torch.cat([neighbor_emb, center_expanded], dim=-1)

        # Pass through MLP
        # Flatten: (B * N, 2D) -> MLP -> (B * N, EventDim) -> Reshape
        events = self._likes(inp.view(-1, inp.shape[-1]))
        return events.view(batch_size, self.num_neighbors, -1)

    def _apply_logic(
        self,
        target_event: torch.Tensor,
        u_neighbor_events: torch.Tensor,
        i_neighbor_events: torch.Tensor,
        u_masks: torch.Tensor,
        i_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies the GCR logic: Score = Sim( OR(NOT(n1), ..., NOT(nk), Target), TRUE )
        Ignores padding events completely during the recursive OR aggregation.
        """
        device = target_event.device
        batch_size = target_event.size(0)

        # Apply NOT to neighbors
        u_neighbor_neg = self._not(u_neighbor_events)
        i_neighbor_neg = self._not(i_neighbor_events)

        # Gather all terms [NOT(u_neighbors), NOT(i_neighbors), Target]
        all_terms = torch.cat(
            [u_neighbor_neg, i_neighbor_neg, target_event.unsqueeze(1)], dim=1
        )

        # Gather all masks
        # Target is always valid (True)
        target_mask = torch.ones((batch_size, 1), device=device, dtype=torch.bool)
        all_masks = torch.cat([u_masks, i_masks, target_mask], dim=1)

        # Shuffle terms and masks together using the same permutation
        perm = torch.randperm(all_terms.size(1))
        shuffled_terms = all_terms[:, perm, :]
        shuffled_masks = all_masks[:, perm]

        # Recursive OR aggregation with skipping logic
        # We initialize the accumulator with zeros, but it will be overwritten
        # by the first valid term encountered in the loop.
        logic_accum = torch.zeros_like(target_event)
        initialized_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)

        for k in range(shuffled_terms.size(1)):
            term_k = shuffled_terms[:, k, :]  # (B, D)
            mask_k = shuffled_masks[:, k]  # (B,)

            # Determine the role of this term for each item in the batch:
            # 1. First valid term? -> Initialize accumulator
            is_first = mask_k & (~initialized_mask)

            # 2. Subsequent valid term? -> Perform OR update
            is_update = mask_k & initialized_mask

            # Compute OR update (computed for whole batch, selected via mask later)
            or_out = self._or(logic_accum, term_k)

            # Update accumulator:
            # - If first valid term: set to term_k
            # - If subsequent valid: set to OR(accum, term_k)
            # - If invalid (padding): keep existing logic_accum
            logic_accum = torch.where(
                is_first.unsqueeze(-1),
                term_k,
                torch.where(is_update.unsqueeze(-1), or_out, logic_accum),
            )

            # Update initialization status
            initialized_mask = initialized_mask | mask_k

        # Compute Similarity with TRUE anchor
        return self.sim(logic_accum, self.TRUE)

    def _forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the predicted scores for given user and item IDs.
        """
        device = user_ids.device

        # 1. Embeddings
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        # 2. Encode Target Event e_{u,i}
        target_inputs = torch.cat([u_emb, i_emb], dim=-1)
        target_events = self._likes(target_inputs)

        # 3. Neighbor Sampling
        u_ids_np = user_ids.cpu().numpy()
        i_ids_np = item_ids.cpu().numpy()

        u_neighbor_indices, u_masks = self._sample_neighbors(
            entity_ids=u_ids_np,
            exclusion_ids=i_ids_np,
            interactions_map=self.user_interactions,
            device=device,
        )

        i_neighbor_indices, i_masks = self._sample_neighbors(
            entity_ids=i_ids_np,
            exclusion_ids=u_ids_np,
            interactions_map=self.item_interactions,
            device=device,
        )

        # 4. Compute Neighbor Events
        u_neighbor_events = self._compute_neighbor_events(
            center_emb=u_emb,
            neighbor_indices=u_neighbor_indices,
            neighbor_embedding_layer=self.item_embedding,
            center_is_user=True,
        )

        i_neighbor_events = self._compute_neighbor_events(
            center_emb=i_emb,
            neighbor_indices=i_neighbor_indices,
            neighbor_embedding_layer=self.user_embedding,
            center_is_user=False,
        )

        # 5. Cache events for regularizers (Filter out padding!)
        # u_masks is boolean, so we can index directly
        valid_u_events = u_neighbor_events[u_masks]
        valid_i_events = i_neighbor_events[i_masks]

        self._cached_event_embeddings = torch.cat(
            [target_events, valid_u_events, valid_i_events], dim=0
        )

        # 6. Apply Logical Operations & Return Score
        return self._apply_logic(
            target_events,
            u_neighbor_events,
            i_neighbor_events,
            u_masks,
            i_masks,
        )


class GCRFactory(RecommenderFactory):
    """
    Builder class for the GCR recommender model.
    """

    @property
    def argparser(self) -> ArgumentParser:
        parser = super().argparser
        parser.add_argument(
            "--user-item-embedding-dim",
            type=int,
            required=True,
            help="Dimensionality of the user and item embeddings.",
        )
        parser.add_argument(
            "--event-embedding-dim",
            type=int,
            required=True,
            help="Dimensionality of the event embeddings.",
        )
        parser.add_argument(
            "--hidden-dim",
            type=int,
            required=True,
            help="Dimensionality of the hidden layer in the MLP modules.",
        )
        parser.add_argument(
            "--num-neighbors",
            type=int,
            required=True,
            help="Number of neighbors to consider for each user and for each item.",
        )
        parser.add_argument(
            "--reg-weight",
            type=float,
            required=True,
            help="Regularization weight for the logical modules.",
        )
        return parser

    def create(self, context: Context, args: dict) -> Recommender:
        user_item_embedding_dim = args["user_item_embedding_dim"]
        event_embedding_dim = args["event_embedding_dim"]
        hidden_dim = args["hidden_dim"]
        num_neighbors = args["num_neighbors"]
        reg_weight = args["reg_weight"]
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
