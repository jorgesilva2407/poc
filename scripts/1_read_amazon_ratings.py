import os
import pickle as pkl
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def parse_args():
    parser = ArgumentParser(description="Process Amazon Ratings Dataset")
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Path to the input JSONL file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the processed files",
    )
    parser.add_argument(
        "-m",
        "--min-user-ratings",
        type=int,
        default=3,
        help="Minimum number of ratings per user to keep",
    )
    parsed_args = parser.parse_args()
    return parsed_args.input_file, parsed_args.output_dir, parsed_args.min_user_ratings


def process_dataset(jsonl_file, output_dir, min_user_ratings=3):
    """
    Process a single dataset: read, filter, index, and save.

    Args:
        jsonl_file: Path to the input JSONL file
        output_dir: Directory where processed files will be saved
        min_user_ratings: Minimum number of ratings per user to keep
    """
    # Read ratings
    df = pd.read_json(jsonl_file, lines=True)
    df = df[["reviewerID", "asin", "overall", "unixReviewTime"]]
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    df = df[df["rating"] >= 4]
    df.drop(columns=["rating"], inplace=True)

    # Filter ratings
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    df = df[df["user_id"].isin(valid_users)]

    # Index columns
    user_encoder = LabelEncoder()
    df["user_id"] = user_encoder.fit_transform(df["user_id"])

    item_encoder = LabelEncoder()
    df["item_id"] = item_encoder.fit_transform(df["item_id"])

    # Save outputs
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(output_path / "all_interactions.csv", index=False)

    with open(output_path / "user_encoder.pkl", "wb") as f:
        pkl.dump(user_encoder, f)

    with open(output_path / "item_encoder.pkl", "wb") as f:
        pkl.dump(item_encoder, f)

    print(f"Processed {jsonl_file}")
    print(f"  - Total interactions: {len(df)}")
    print(f"  - Unique users: {df['user_id'].nunique()}")
    print(f"  - Unique items: {df['item_id'].nunique()}")
    print(f"  - Saved to: {output_dir}")


def main() -> None:
    input_file, output_dir, min_user_ratings = parse_args()

    process_dataset(
        jsonl_file=input_file,
        output_dir=output_dir,
        min_user_ratings=min_user_ratings,
    )


if __name__ == "__main__":
    main()
