"""Dataset for precomputed test samples in recommendation systems."""

import torch
import pandas as pd
from src.datasets.recommendation_dataset import TripletSample, RecommendationDataset


class PrecomputedTestDataset(RecommendationDataset):
    """
    Dataset for precomputed test samples in recommendation systems.
    Each sample consists of a user, a positive item, and multiple negative items.
    Args:
        test_df (pd.DataFrame): DataFrame containing precomputed test samples with columns
            'user_id', 'pos_id', 'neg_id_01' ... 'neg_id_99'.
    """

    def __init__(self, test_df: pd.DataFrame):
        self.test_df = test_df

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx) -> TripletSample:
        row = self.test_df.iloc[idx]
        user = torch.tensor(row["user_id"], dtype=torch.long)
        pos_item = torch.tensor(row["pos_id"], dtype=torch.long)
        neg_items = torch.tensor(row.filter(like="neg_id_").values, dtype=torch.long)
        return user, pos_item, neg_items
