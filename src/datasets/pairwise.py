import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class PairwiseDataset(Dataset):
    def __init__(self, df):
        self.user_pos = (
            df[df["label"] == 1].groupby("user_id")["item_id"].apply(set).to_dict()
        )
        self.user_neg = (
            df[df["label"] == 0].groupby("user_id")["item_id"].apply(set).to_dict()
        )

        self.users = list(self.user_pos.keys())
        self.items = set(df["item_id"].unique())

        self.size = len(self.users)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_items = list(self.user_pos[user])
        neg_items = list(self.user_neg.get(user, []))
        if len(neg_items) == 0:
            neg_items = list(self.items - self.user_pos[user])

        pos_item = np.random.choice(pos_items)
        neg_item = np.random.choice(neg_items)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


def main():
    df = pd.read_csv("data/processed/train.csv")
    dataset = PairwiseDataset(df)
    print(f"Dataset size: {len(dataset)}")
    for _ in range(5):
        user, pos_item, neg_item = dataset[np.random.randint(0, len(dataset))]
        print(f"User: {user}, Positive Item: {pos_item}, Negative Item: {neg_item}")


if __name__ == "__main__":
    main()
