"""Feature-Aware Encoder Module."""

from typing import Literal

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
        num_entities: int,
        feature_path: str,
        feature_metadata: dict[str, Literal["boolean", "numerical", "categorical"]],
        categorical_embedding_dim: int,
        embed_id: bool,
        output_dim: int,
        dropout_rate: float,
    ):
        """
        Encodes entities by concatenating their ID embedding (optional)
        with embeddings of their categorical features and raw numerical features.

        Args:
            feature_path (str): Path to the CSV file containing features (e.g., users.csv).
            feature_metadata (dict): Dictionary mapping column names to types.
            categorical_embedding_dim (int): Dimension size for categorical embeddings.
            embed_id (bool): Whether to learn a specific embedding for the Entity ID itself.
            output_dim (int): Projects the concatenated vector to this size.
            dropout_rate (float): Dropout rate applied after projection (if any).
        """
        super().__init__()

        self._categorical_embedding_dim = categorical_embedding_dim

        # Calculate the size of the concatenated vector
        concat_dim = 0

        # 1. Load Data
        # We assume the CSV fits in memory (ML-1M is small).
        df = pd.read_csv(feature_path)

        # Identify the ID column (keys ending in '_id' based on your prep script)
        id_col = next(col for col in feature_metadata.keys() if col.endswith("_id"))

        # Ensure the dataframe is sorted by ID so array index matches Entity ID
        # This is crucial because we will look up features by tensor index.
        df = df.sort_values(by=id_col).reset_index(drop=True)

        # 2. Setup ID Embedding
        self.use_id_embedding = embed_id
        if self.use_id_embedding:
            self.id_embedding = nn.Embedding(num_entities, categorical_embedding_dim)
            concat_dim += categorical_embedding_dim

        # 3. Setup Feature Layers
        self.categorical_embeddings = nn.ModuleDict()

        # Buffers to hold static feature data on device
        # We separate categorical indices and numerical values
        self.cat_feature_cols = []
        self.num_feature_cols = []

        for col, dtype in feature_metadata.items():
            # Skip the main ID column here, processed above
            if col == id_col:
                continue

            if dtype == "categorical":
                # Create embedding layer for this feature
                # Cardinality: max value + 1
                cardinality = df[col].max() + 1
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
        embeddings = []

        # 1. Main Entity ID Embedding
        if self.use_id_embedding:
            embeddings.append(self.id_embedding(ids))

        # 2. Categorical Features
        for col in self.cat_feature_cols:
            # Retrieve indices from buffer: self.data_{col}[ids]
            feature_indices = getattr(self, f"data_{col}")[ids]
            # Pass through specific embedding layer
            emb = self.categorical_embeddings[col](feature_indices)
            embeddings.append(emb)

        # 3. Numerical/Boolean Features
        for col in self.num_feature_cols:
            # Retrieve values from buffer: self.data_{col}[ids]
            values = getattr(self, f"data_{col}")[ids]
            # Unsqueeze to make it (Batch, 1) so it can be concatenated
            embeddings.append(values.unsqueeze(1))

        # 4. Concatenate all parts
        if not embeddings:
            raise ValueError("Encoder has no features enabled and embed_id is False.")

        concat_vector = torch.cat(embeddings, dim=1)

        # 5. Project to output dimension
        return self.projection(concat_vector)
