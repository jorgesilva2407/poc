import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class PointwiseDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user = row["user_id"]
        item = row["item_id"]
        label = row["label"]

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )


def main():
    df = pd.read_csv("data/processed/train.csv")
    dataset = PointwiseDataset(df)
    print(f"Dataset size: {len(dataset)}")
    for _ in range(5):
        user, item, label = dataset[np.random.randint(0, len(dataset))]
        print(f"User: {user}, Item: {item}, Label: {label}")


if __name__ == "__main__":
    main()
