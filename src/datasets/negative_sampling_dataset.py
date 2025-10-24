import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import Dataset

TripletSample = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class NegativeSamplingDataset(Dataset[TripletSample]):
    def __init__(
        self,
        interactions: pd.DataFrame,
        all_interactions: pd.DataFrame,
        num_negatives: int,
    ):
        self.num_negatives = num_negatives

        self.user_train_interactions = (
            interactions.groupby("user_id")["item_id"].apply(set).to_dict()
        )
        self.user_all_interactions = (
            all_interactions.groupby("user_id")["item_id"].apply(set).to_dict()
        )

        self.users = list(self.user_train_interactions.keys())
        self.items = set(all_interactions["item_id"].unique())

        self.size = len(self.users)

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> TripletSample:
        user = self.users[idx]

        pos_items = list(self.user_train_interactions[user])
        neg_items = list(self.items - self.user_all_interactions[user])

        pos_item = np.random.choice(pos_items)
        neg_item = np.random.choice(neg_items, size=self.num_negatives, replace=False)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


def parse_args():
    parser = ArgumentParser(description="Pairwise Dataset Example")
    parser.add_argument(
        "-i",
        "--interactions-data-path",
        type=str,
        default="data/processed/amazon-2014/split/train.csv",
        help="Path to the interactions CSV file",
    )
    parser.add_argument(
        "-a",
        "--all-data-path",
        type=str,
        default="data/processed/amazon-2014/all_interactions.csv",
        help="Path to the all interactions CSV file",
    )
    parsed_args = parser.parse_args()
    return parsed_args.interactions_data_path, parsed_args.all_data_path


def main():
    interactions_data_path, all_data_path = parse_args()
    train_interactions = pd.read_csv(interactions_data_path)
    all_interactions = pd.read_csv(all_data_path)
    dataset = NegativeSamplingDataset(train_interactions, all_interactions, 5)
    print(f"Dataset size: {len(dataset)}")
    for _ in range(5):
        user, pos_item, neg_item = dataset[np.random.randint(0, len(dataset))]
        print(f"User: {user}, Positive Item: {pos_item}, Negative Item: {neg_item}")


if __name__ == "__main__":
    main()
