import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from src.datasets.negative_sampling_dataset import NegativeSamplingDataset


def parse_args():
    parser = ArgumentParser(description="Precompute Test Negative Samples")
    parser.add_argument(
        "-s",
        "--split-data-dir",
        type=str,
        default="data/processed/amazon-2014/split/",
        help="Path to the split data directory",
    )
    parser.add_argument(
        "-a",
        "--all-data-path",
        type=str,
        default="data/processed/amazon-2014/all_interactions.csv",
        help="Path to the all interactions CSV file",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="data/processed/amazon-2014/split/test_neg_samples.csv",
        help="Path to save the precomputed test negative samples",
    )
    parser.add_argument(
        "-n",
        "--num-negatives",
        type=int,
        default=99,
        help="Number of negative samples to precompute per user",
    )
    parser.add_argument(
        "-r",
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parsed_args = parser.parse_args()
    return (
        parsed_args.split_data_dir,
        parsed_args.all_data_path,
        parsed_args.output_path,
        parsed_args.num_negatives,
        parsed_args.random_seed,
    )


def main():
    (
        split_data_dir,
        all_data_path,
        output_path,
        num_negatives,
        random_seed,
    ) = parse_args()

    test_data_path = os.path.join(split_data_dir, "test.csv")
    test_interactions = pd.read_csv(test_data_path)
    all_interactions = pd.read_csv(all_data_path)

    dataset = NegativeSamplingDataset(
        test_interactions,
        all_interactions,
        num_negatives,
    )

    np.random.seed(random_seed)

    rows = []
    for idx in tqdm(range(len(dataset)), desc="Processing"):
        user, pos_item, neg_items = dataset[idx]
        user_id = user.item()
        pos_id = pos_item.item()
        neg_item_ids = neg_items.tolist()
        row = [user_id, pos_id] + neg_item_ids
        rows.append(row)

    pad_width = len(str(num_negatives))
    neg_columns = [f"neg_id_{i+1:0{pad_width}d}" for i in range(num_negatives)]
    columns = ["user_id", "pos_id"] + neg_columns

    neg_samples_df = pd.DataFrame(rows, columns=columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    neg_samples_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
