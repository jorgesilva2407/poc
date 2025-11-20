"""Interfaces for GCR components to ensure type safety."""

from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.recommender import Recommender, RecommenderFactory
from src.models.logical_modules.interfaces import LogicalOR, LogicalNOT
from src.models.encoders.interface import Encoder


class BaseGCR(Recommender):
    """
    Abstract Base Class for Graph Collaborative Reasoning models.

    Shared Components:
    - Neighbor sampling logic.
    - Event Encoder (_likes) with Dynamic Input.
    - Anchor Vector (TRUE).
    - Reasoning orchestration (_forward, _apply_logic).

    Dependency Injection:
    - user_encoder: Encoder implementation
    - item_encoder: Encoder implementation
    - or_module: LogicalOR implementation
    - not_module: LogicalNOT implementation
    """

    def __init__(
        self,
        name: str,
        num_users: int,
        num_items: int,
        interactions: pd.DataFrame,
        user_encoder: Encoder,
        item_encoder: Encoder,
        or_module: LogicalOR,
        not_module: LogicalNOT,
        event_embedding_dim: int,
        hidden_dim: int,
        num_neighbors: int,
        should_permute: bool,  # Whether to permute the logical terms (works as regularizer)
    ):
        super().__init__(name, num_users, num_items)
        self.event_embedding_dim = event_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors
        self.should_permute = should_permute

        # Injected Encoders
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder

        # Injected Logic Modules
        self._or = or_module
        self._not = not_module

        # Cache for regularizers (optional, used if subclass implements hooks)
        self._cached_event_embeddings = None

        # 1. Shared Anchor Vector (TRUE)
        self.register_buffer("TRUE", torch.rand(event_embedding_dim))

        # 2. Shared Event Encoder (_likes)
        # Projects (User, Item) pairs into the Event Space.
        # We calculate the input dimension from the injected encoders.
        input_dim = self.user_encoder.embedding_dim + self.item_encoder.embedding_dim

        self._likes = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, event_embedding_dim),
        )

        # Precompute interactions for sampling
        self.user_interactions = (
            interactions.groupby("user_id")["item_id"]
            .apply(lambda x: np.array(x.unique(), dtype=np.int64))
            .to_dict()
        )

        self.item_interactions = (
            interactions.groupby("item_id")["user_id"]
            .apply(lambda x: np.array(x.unique(), dtype=np.int64))
            .to_dict()
        )

    def sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute similarity. Can be overridden for Fuzzy Logic if needed."""
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        # pylint: disable=not-callable
        return F.cosine_similarity(a, b, dim=-1)
        # pylint: enable=not-callable

    @property
    @abstractmethod
    def hparams(self) -> dict[str, int]:
        return {
            "event_emb_dim": self.event_embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_neighbors": self.num_neighbors,
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
                # Take all valid neighbors, pad remainder with zeros
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
        masks: torch.Tensor,
        encoder: Encoder,
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

        # Handle padding safely using the mask for lookup
        safe_indices = neighbor_indices.clone()
        safe_indices[~masks] = 0

        # Retrieve neighbor embeddings via injected encoder
        neighbor_emb = encoder(safe_indices)

        # Concatenate based on order required by _likes (User, Item)
        if center_is_user:
            # Center is User, Neighbor is Item -> (User, NeighborItem)
            inp = torch.cat([center_expanded, neighbor_emb], dim=-1)
        else:
            # Center is Item, Neighbor is User -> (NeighborUser, Item)
            inp = torch.cat([neighbor_emb, center_expanded], dim=-1)

        # Pass through MLP (Event Encoder)
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

        # Apply NOT to neighbor
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
        # Only permute if enabled
        if self.should_permute:
            perm = torch.randperm(all_terms.size(1))
            shuffled_terms = all_terms[:, perm, :]
            shuffled_masks = all_masks[:, perm]
        else:
            shuffled_terms = all_terms
            shuffled_masks = all_masks

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

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        The main orchestration method.
        """
        device = user_ids.device

        # 1. Embeddings (Via Injected Encoders)
        u_emb = self.user_encoder(user_ids)
        i_emb = self.item_encoder(item_ids)

        # 2. Encode Target Event e_{u,i}
        target_input = torch.cat([u_emb, i_emb], dim=-1)
        target_event = self._likes(target_input)

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
            masks=u_masks,
            encoder=self.item_encoder,
            center_is_user=True,
        )

        i_neighbor_events = self._compute_neighbor_events(
            center_emb=i_emb,
            neighbor_indices=i_neighbor_indices,
            masks=i_masks,
            encoder=self.user_encoder,
            center_is_user=False,
        )

        # 5. Cache events for regularizers (if hooks are used) (Filter out padding!)
        valid_u_events = u_neighbor_events[u_masks]
        valid_i_events = i_neighbor_events[i_masks]

        self._cached_event_embeddings = torch.cat(
            [target_event, valid_u_events, valid_i_events], dim=0
        )

        # 6. Apply Logical Operations & Return Score
        return self._apply_logic(
            target_event, u_neighbor_events, i_neighbor_events, u_masks, i_masks
        )


class BaseGCRFactory(RecommenderFactory):
    """
    Factory Interface for creating BaseGCR instances.
    Ensures correct dependency injection and configuration.
    """

    @property
    def argparser(self):
        """Return argument parser for GCR model configuration."""
        parser = super().argparser
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
        return parser
