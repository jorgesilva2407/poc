"""Feature-Aware Encoder Module."""

from typing import Dict, Literal

import torch
import torch.nn as nn
import pandas as pd

from src.models.encoders.interface import Encoder


class FeatureAwareEncoder(Encoder):
    """
    Encodes entities by concatenating their ID embedding (optional)
    with embeddings of their categorical features and raw numerical features.
    """

    def __init__(
        self,
        id_col: str,
        num_entities: int,
        feature_path: str,
        feature_metadata: Dict[str, Literal["boolean", "numerical", "categorical"]],
        categorical_embedding_dim: int,
        embed_id: bool,
        output_dim: int,
        dropout_rate: float,
    ):
        """
        Encodes entities by concatenating their ID embedding (optional)
        with embeddings of their categorical features and raw numerical features.

        Args:
            num_entities (int): Total number of entities (used for ID embedding).
            feature_path (str): Path to the CSV file containing features.
            feature_metadata (dict): Dictionary mapping column names to types.
            categorical_embedding_dim (int): Dimension size for categorical embeddings.
            embed_id (bool): Whether to learn a specific embedding for the Entity ID itself.
            output_dim (int): Projects the concatenated vector to this size.
            dropout_rate (float): Dropout rate applied after projection.
        """
        super().__init__()

        self._categorical_embedding_dim = categorical_embedding_dim

        # Calculate the size of the concatenated vector
        concat_dim = 0

        # 1. Load Data
        # We assume the CSV fits in memory.
        df = pd.read_csv(feature_path)
        df = df.sort_values(by=id_col).reset_index(drop=True)

        # 2. Setup ID Embedding
        self.use_id_embedding = embed_id
        if self.use_id_embedding:
            self.id_embedding = nn.Embedding(num_entities, categorical_embedding_dim)
            concat_dim += categorical_embedding_dim

        # 3. Setup Feature Layers
        self.categorical_embeddings = nn.ModuleDict()

        # Buffers to hold static feature data on device
        self.cat_feature_cols = []
        self.num_feature_cols = []

        for col, dtype in feature_metadata.items():
            # Skip the main ID column here, processed above
            if col == id_col:
                continue

            if col not in df.columns:
                # Warn or skip if metadata has col not in DF, but here we skip strictly
                continue

            if dtype == "categorical":
                # Create embedding layer for this feature
                # Cardinality: max value + 1 (assuming 0-indexed encoded integers)
                cardinality = int(df[col].max()) + 1
                self.categorical_embeddings[col] = nn.Embedding(
                    cardinality, categorical_embedding_dim
                )
                self.cat_feature_cols.append(col)
                concat_dim += categorical_embedding_dim

                # Register data as buffer (shape: [N_entities])
                self.register_buffer(
                    f"data_{col}", torch.tensor(df[col].values, dtype=torch.long)
                )

            elif dtype in ["numerical", "boolean"]:
                self.num_feature_cols.append(col)
                # Numerical/Boolean adds 1 dimension per column
                concat_dim += 1

                # Register data as buffer (shape: [N_entities])
                self.register_buffer(
                    f"data_{col}", torch.tensor(df[col].values, dtype=torch.float32)
                )

        # 4. Setup Projection (Dimensionality Reduction)
        self._output_dim = output_dim
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    @property
    def embedding_dim(self) -> int:
        return self._output_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Constructs the embedding vector for the given entity IDs.
        Order: [ID_Embed, Cat_Embed_1, Cat_Embed_2, ..., Num_Feat_1, Num_Feat_2...]
        """
        # Store original shape to restore later (e.g., [Batch, Neighbors])
        original_shape = ids.shape

        # Flatten ids for feature lookup (requires 1D index)
        ids_flat = ids.flatten()

        # Clamp IDs to ensure they don't exceed loaded feature length
        # (Optional safety, helps if unexpected IDs appear)
        # ids_flat = ids_flat.clamp(0, self.data_norm_age.shape[0] - 1)

        embeddings = []

        # 1. Main Entity ID Embedding
        if self.use_id_embedding:
            embeddings.append(self.id_embedding(ids_flat))

        # 2. Categorical Features
        for col in self.cat_feature_cols:
            # Retrieve indices from buffer
            # We must use getattr because buffers are registered dynamically
            feature_indices = getattr(self, f"data_{col}")[ids_flat]

            # Pass through specific embedding layer
            emb = self.categorical_embeddings[col](feature_indices)
            embeddings.append(emb)

        # 3. Numerical/Boolean Features
        for col in self.num_feature_cols:
            # Retrieve values from buffer
            values = getattr(self, f"data_{col}")[ids_flat]

            # Unsqueeze to make it (Batch*K, 1) so it can be concatenated with embeddings
            embeddings.append(values.unsqueeze(1))

        # 4. Concatenate all parts
        if not embeddings:
            raise ValueError("Encoder has no features enabled and embed_id is False.")

        concat_vector = torch.cat(embeddings, dim=1)

        # 5. Project to output dimension
        output = self.projection(concat_vector)

        # 6. Restore original shape (CRITICAL FIX FOR GCR)
        # If input was [B, K], output becomes [B, K, Output_Dim]
        if len(original_shape) > 1:
            output = output.view(*original_shape, -1)

        return output
